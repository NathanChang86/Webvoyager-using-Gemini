import os
import re
import time
import json
import base64
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from AR_Agent_prompts import SYSTEM_PROMPT
import google.generativeai as gemini
from google.generativeai.types import GenerationConfig



# 設置log
def setup_logger(folder_path):
    log_file_path = os.path.join(folder_path, 'agent.log')

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # 創建文件處理器 (FileHandler)，並設置其寫入文件的路徑
    # 在這裡加上 encoding='utf-8' 來確保能處理中文字符
    handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 使用者操作紀錄 ------------------------------------------------------
def driver_config():
    options = webdriver.ChromeOptions()
    options.add_experimental_option(
        "prefs", {
            "plugins.always_open_pdf_externally": True
        }
    )
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    options.add_argument("disable-blink-features=AutomationControlled")
    return options

def inject_script(driver_task):
    driver_task.execute_script("""
        (function() {
            function setupUserInteractionListener() {
                if (window.__userInteractionInjected__ && window.__lastInjectedBody__ === document.body) {
                    console.log('[INFO] Script already injected, skipping.');
                    return;
                }
                console.log('[INFO] Injecting interaction listeners.');
                
                // 移除舊監聽器
                ['click', 'input', 'scroll', 'popstate', 'pageshow', 'beforeunload', 'okPrompt', 'change'].forEach(type => {
                    let listener = window[`__${type}Listener`];
                    if (listener) {
                        if (type === 'popstate' || type === 'pageshow') {
                            window.removeEventListener(type, listener);
                        } else {
                            document.removeEventListener(type, listener, true);
                        }
                        console.log(`[INFO] Removed existing ${type} listener.`);
                    }
                });

                window.__userInteractionInjected__ = true;
                window.__lastInjectedBody__ = document.body;

                window.userInteractions = JSON.parse(sessionStorage.getItem('userInteractions') || '[]');
                let lastRecordedValue = '';
                let lastClickTimestamp = 0;
                window.__processingPopstate__ = false;
                let lastScrollY = window.scrollY;

                // *** 修改部分 1：優化 clickListener，完全避免干擾 <select> ***
                const clickListener = function(event) {
                    if (event.__processed__) return;
                    const now = new Date().getTime();
                    if (now - lastClickTimestamp < 100) return;

                    // 檢查是否為下拉選單相關元素
                    const isDropdown = event.target.closest('select, option, [role="listbox"], [role="option"], .dropdown, .dropdown-menu');
                    if (isDropdown) {
                        // 記錄 select/option 的點擊，但不干擾事件
                        if (event.target.tagName === 'SELECT' || event.target.tagName === 'OPTION') {
                            let target = event.target;
                            let interaction = {
                                type: 'click',
                                target: target.tagName,
                                id: target.id,
                                class: target.className,
                                text: (target.innerText || target.value || '').slice(0, 100),
                                x: event.clientX,
                                y: event.clientY
                            };
                            window.userInteractions.push(interaction);
                            sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                        }
                        return; // 不做任何處理，確保下拉選單正常展開
                    }

                    lastClickTimestamp = now;
                    event.__processed__ = true;

                    let target = event.target.closest('a, img, [onclick], [data-nav], [role="option"], li, div, span') || event.target;
                    let rect = target.getBoundingClientRect();
                    let interaction = {
                        type: 'click',
                        target: target.tagName,
                        id: target.id,
                        class: target.className,
                        text: (target.innerText || target.value || '').slice(0, 100),
                        x: event.clientX,
                        y: event.clientY
                    };

                    const currentUrl = window.location.href.split('#')[0];
                    const targetUrl = (target.href || '').split('#')[0];

                    if (target.tagName.toLowerCase() === 'a' && target.href) {
                        if (currentUrl === targetUrl) {
                            window.userInteractions.push(interaction);
                            sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                        } else {
                            event.preventDefault();
                            window.userInteractions.push(interaction);
                            sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                            let clicks = JSON.parse(localStorage.getItem('navigationClicks') || '[]');
                            clicks.push(interaction);
                            localStorage.setItem('navigationClicks', JSON.stringify(clicks));
                            setTimeout(() => {
                                window.location.href = target.href;
                            }, 300);
                        }
                    } else if (target.closest('form') && window.location.href.includes('google.com')) {
                        window.userInteractions.push(interaction);
                        sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                        event.preventDefault();
                        setTimeout(() => {
                            target.closest('form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
                        }, 300);
                    } else {
                        window.userInteractions.push(interaction);
                        sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                    }
                };
                window.__clickListener = clickListener;
                document.addEventListener('click', clickListener, false);

                // 添加對綠框覆蓋層的點擊監聽
                const okPromptListener = function(event) {
                    if (event.target.id === 'ok_prompt_overlay') {
                        window.exitInteractionLoop = true;
                        event.stopPropagation();
                    }
                };
                window.__okPromptListener = okPromptListener;
                document.addEventListener('click', okPromptListener, true);

                const inputListener = function(event) {
                    if ((event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') && event.target.value !== lastRecordedValue) {
                        lastRecordedValue = event.target.value;
                        let interaction = {
                            type: 'input',
                            value: event.target.value.slice(0, 100),
                            target: event.target.tagName,
                            id: event.target.id,
                            class: event.target.className
                        };
                        window.userInteractions.push(interaction);
                        sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                    }
                };
                window.__inputListener = inputListener;
                document.addEventListener('input', inputListener, true);

                const scrollListener = function() {
                    if (window.scrollY !== lastScrollY) {
                        let interaction = {
                            type: 'scroll',
                            scrollY: window.scrollY
                        };
                        window.userInteractions.push(interaction);
                        sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                        lastScrollY = window.scrollY;
                    }
                };
                window.__scrollListener = scrollListener;
                window.addEventListener('scroll', scrollListener);

                const changeListener = function(event) {
                    if (event.target.tagName === 'SELECT') {
                        let interaction = {
                            type: 'change',
                            target: event.target.tagName,
                            id: event.target.id,
                            class: event.target.className,
                            value: event.target.value.slice(0, 100),
                            selectedText: (event.target.selectedOptions[0]?.text || '').slice(0, 100)
                        };
                        window.userInteractions.push(interaction);
                        sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                    }
                };
                window.__changeListener = changeListener;
                document.addEventListener('change', changeListener, false);

                const popstateListener = function(event) {
                    if (window.__processingPopstate__) {
                        console.log('[INFO] Ignoring popstate event during processing.');
                        return;
                    }
                    window.__processingPopstate__ = true;
                    let interaction = {
                        type: 'navigation',
                        action: 'back',
                        url: window.location.href
                    };
                    if (!window.userInteractions.some(i => i.type === 'navigation' && i.url === interaction.url && Math.abs(new Date(i.timestamp) - new Date(interaction.timestamp)) < 100)) {
                        window.userInteractions.push(interaction);
                        sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                        localStorage.setItem('pendingPopstateInteraction', JSON.stringify(interaction));
                        console.log('[INFO] Popstate interaction stored:', interaction);
                    }
                    setTimeout(() => {
                        window.__userInteractionInjected__ = false;
                        setupUserInteractionListener();
                    }, 300);
                };
                window.__popstateListener = popstateListener;
                window.addEventListener('popstate', popstateListener);

                const pageshowListener = function(event) {
                    if (event.persisted) {
                        console.log('[INFO] Page restored from bfcache, reinjecting.');
                        let interaction = {
                            type: 'navigation',
                            action: 'back',
                            url: window.location.href
                        };
                        if (!window.userInteractions.some(i => i.type === 'navigation' && i.url === interaction.url && Math.abs(new Date(i.timestamp) - new Date(interaction.timestamp)) < 100)) {
                            window.userInteractions.push(interaction);
                            sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                            localStorage.setItem('pendingPopstateInteraction', JSON.stringify(interaction));
                            console.log('[INFO] Pageshow interaction stored:', interaction);
                        }
                        window.__userInteractionInjected__ = false;
                        setTimeout(() => {
                            setupUserInteractionListener();
                        }, 300);
                    }
                };
                window.__pageshowListener = pageshowListener;
                window.addEventListener('pageshow', pageshowListener);

                const beforeunloadListener = function() {
                    sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                    localStorage.setItem('navigationClicks', JSON.stringify([]));
                };
                window.__beforeunloadListener = beforeunloadListener;
                window.addEventListener('beforeunload', beforeunloadListener);

                (function recoverNavigation() {
                    const navClicks = JSON.parse(localStorage.getItem('navigationClicks') || '[]');
                    const pendingPopstate = JSON.parse(localStorage.getItem('pendingPopstateInteraction') || 'null');
                    const currentUrl = window.location.href;
                    if (pendingPopstate && pendingPopstate.url !== currentUrl) {
                        window.userInteractions.push(pendingPopstate);
                        sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                        localStorage.removeItem('pendingPopstateInteraction');
                    }
                    if (navClicks.length > 0) {
                        const lastClick = navClicks.pop();
                        window.userInteractions.push(lastClick);
                        sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
                        localStorage.setItem('navigationClicks', JSON.stringify(navClicks));
                    }
                })();
            }

            function monkeyPatchHistoryMethod(methodName) {
                const original = history[methodName];
                history[methodName] = function() {
                    const result = original.apply(this, arguments);
                    setTimeout(() => {
                        window.__userInteractionInjected__ = false;
                        setupUserInteractionListener();
                    }, 300);
                    return result;
                };
            }

            setupUserInteractionListener();
            monkeyPatchHistoryMethod('pushState');
            monkeyPatchHistoryMethod('replaceState');

            window.addEventListener('load', function() {
                setTimeout(() => {
                    window.__userInteractionInjected__ = false;
                    setupUserInteractionListener();
                }, 300);
            });

            // *** 修改部分 2：優化 MutationObserver，避免不必要重新注入 ***
            const observer = new MutationObserver((mutations) => {
                for (const mutation of mutations) {
                    if (mutation.type === 'childList' && mutation.target === document.documentElement) {
                        if (window.__lastInjectedBody__ !== document.body) {
                            window.__userInteractionInjected__ = false;
                            setupUserInteractionListener();
                            break;
                        }
                    }
                }
            });
            observer.observe(document.documentElement, { childList: true, subtree: false });
        })();
    """)

# def inject_script(driver_task):
#     driver_task.execute_script("""
#         (function() {
#             function setupUserInteractionListener() {
#                 if (window.__userInteractionInjected__ && window.__lastInjectedBody__ === document.body) {
#                     console.log('[INFO] Script already injected, skipping.');
#                     return;
#                 }
#                 console.log('[INFO] Injecting interaction listeners.');
                
#                 // 移除舊監聽器
#                 ['click', 'input', 'scroll', 'popstate', 'pageshow', 'beforeunload', 'okPrompt'].forEach(type => {
#                     let listener = window[`__${type}Listener`];
#                     if (listener) {
#                         if (type === 'popstate' || type === 'pageshow') {
#                             window.removeEventListener(type, listener);
#                         } else {
#                             document.removeEventListener(type, listener, true);
#                         }
#                         console.log(`[INFO] Removed existing ${type} listener.`);
#                     }
#                 });

#                 window.__userInteractionInjected__ = true;
#                 window.__lastInjectedBody__ = document.body;

#                 window.userInteractions = JSON.parse(sessionStorage.getItem('userInteractions') || '[]');
#                 let lastRecordedValue = '';
#                 let lastClickTimestamp = 0;
#                 window.__processingPopstate__ = false;
#                 let lastScrollY = window.scrollY;

#                 const clickListener = function(event) {
#                     if (event.__processed__) return;
#                     const now = new Date().getTime();
#                     if (now - lastClickTimestamp < 50) return;
#                     lastClickTimestamp = now;
#                     event.__processed__ = true;

#                     let target = event.target.closest('a, img, [onclick], [data-nav], [role="option"], li, div, span') || event.target;
#                     let rect = target.getBoundingClientRect();
#                     let interaction = {
#                         type: 'click',
#                         target: target.tagName,
#                         id: target.id,
#                         class: target.className,
#                         text: (target.innerText || target.value || '').slice(0, 100),
#                         x: event.clientX,
#                         y: event.clientY
#                     };

#                     const currentUrl = window.location.href.split('#')[0];
#                     const targetUrl = (target.href || '').split('#')[0];

#                     if (target.tagName.toLowerCase() === 'a' && target.href) {
#                         if (currentUrl === targetUrl) {
#                             window.userInteractions.push(interaction);
#                             sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
#                         } else {
#                             event.preventDefault();
#                             window.userInteractions.push(interaction);
#                             sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
#                             let clicks = JSON.parse(localStorage.getItem('navigationClicks') || '[]');
#                             clicks.push(interaction);
#                             localStorage.setItem('navigationClicks', JSON.stringify(clicks));
#                             setTimeout(() => {
#                                 window.location.href = target.href;
#                             }, 300);
#                         }
#                     } else if (target.closest('form') && window.location.href.includes('google.com')) {
#                         window.userInteractions.push(interaction);
#                         sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
#                         event.preventDefault();
#                         setTimeout(() => {
#                             target.closest('form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
#                         }, 300);
#                     } else {
#                         window.userInteractions.push(interaction);
#                         sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
#                     }
#                 };
#                 window.__clickListener = clickListener;
#                 document.addEventListener('click', clickListener, true);

#                 // 添加對綠框覆蓋層的點擊監聽
#                 const okPromptListener = function(event) {
#                     if (event.target.id === 'ok_prompt_overlay') {
#                         window.exitInteractionLoop = true;
#                         event.stopPropagation();
#                     }
#                 };
#                 window.__okPromptListener = okPromptListener;
#                 document.addEventListener('click', okPromptListener, true);

#                 const inputListener = function(event) {
#                     if ((event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') && event.target.value !== lastRecordedValue) {
#                         lastRecordedValue = event.target.value;
#                         let interaction = {
#                             type: 'input',
#                             value: event.target.value.slice(0, 100),
#                             target: event.target.tagName,
#                             id: event.target.id,
#                             class: event.target.className
#                         };
#                         window.userInteractions.push(interaction);
#                         sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
#                     }
#                 };
#                 window.__inputListener = inputListener;
#                 document.addEventListener('input', inputListener, true);

#                 const scrollListener = function() {
#                     if (window.scrollY !== lastScrollY) {
#                         let interaction = {
#                             type: 'scroll',
#                             scrollY: window.scrollY
#                         };
#                         window.userInteractions.push(interaction);
#                         sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
#                         lastScrollY = window.scrollY;
#                     }
#                 };
#                 window.__scrollListener = scrollListener;
#                 window.addEventListener('scroll', scrollListener);

#                 const popstateListener = function(event) {
#                     if (window.__processingPopstate__) {
#                         console.log('[INFO] Ignoring popstate event during processing.');
#                         return;
#                     }
#                     window.__processingPopstate__ = true;
#                     let interaction = {
#                         type: 'navigation',
#                         action: 'back',
#                         url: window.location.href
#                     };
#                     // 避免重複記錄
#                     if (!window.userInteractions.some(i => i.type === 'navigation' && i.url === interaction.url && Math.abs(new Date(i.timestamp) - new Date(interaction.timestamp)) < 100)) {
#                         window.userInteractions.push(interaction);
#                         sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
#                         localStorage.setItem('pendingPopstateInteraction', JSON.stringify(interaction));
#                         console.log('[INFO] Popstate interaction stored:', interaction);
#                     }
#                     setTimeout(() => {
#                         window.__userInteractionInjected__ = false;
#                         setupUserInteractionListener();
#                         window.__processingPopstate__ = false;
#                     }, 300);
#                 };
#                 window.__popstateListener = popstateListener;
#                 window.addEventListener('popstate', popstateListener);

#                 const pageshowListener = function(event) {
#                     if (event.persisted) {
#                         console.log('[INFO] Page restored from bfcache, reinjecting.');
#                         let interaction = {
#                             type: 'navigation',
#                             action: 'back',
#                             url: window.location.href
#                         };
#                         // 避免重複記錄
#                         if (!window.userInteractions.some(i => i.type === 'navigation' && i.url === interaction.url && Math.abs(new Date(i.timestamp) - new Date(interaction.timestamp)) < 100)) {
#                             window.userInteractions.push(interaction);
#                             sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
#                             localStorage.setItem('pendingPopstateInteraction', JSON.stringify(interaction));
#                             console.log('[INFO] Pageshow interaction stored:', interaction);
#                         }
#                         window.__userInteractionInjected__ = false;
#                         setTimeout(() => {
#                             setupUserInteractionListener();
#                         }, 300);
#                     }
#                 };
#                 window.__pageshowListener = pageshowListener;
#                 window.addEventListener('pageshow', pageshowListener);

#                 const beforeunloadListener = function() {
#                     sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
#                     localStorage.setItem('navigationClicks', JSON.stringify([]));
#                 };
#                 window.__beforeunloadListener = beforeunloadListener;
#                 window.addEventListener('beforeunload', beforeunloadListener);

#                 (function recoverNavigation() {
#                     const navClicks = JSON.parse(localStorage.getItem('navigationClicks') || '[]');
#                     const pendingPopstate = JSON.parse(localStorage.getItem('pendingPopstateInteraction') || 'null');
#                     const currentUrl = window.location.href;
#                     if (pendingPopstate && pendingPopstate.url !== currentUrl) {
#                         window.userInteractions.push(pendingPopstate);
#                         sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
#                         localStorage.removeItem('pendingPopstateInteraction');
#                     }
#                     if (navClicks.length > 0) {
#                         const lastClick = navClicks.pop();
#                         window.userInteractions.push(lastClick);
#                         sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
#                         localStorage.setItem('navigationClicks', JSON.stringify(navClicks));
#                     }
#                 })();
#             }

#             function monkeyPatchHistoryMethod(methodName) {
#                 const original = history[methodName];
#                 history[methodName] = function() {
#                     const result = original.apply(this, arguments);
#                     setTimeout(() => {
#                         window.__userInteractionInjected__ = false;
#                         setupUserInteractionListener();
#                     }, 300);
#                     return result;
#                 };
#             }

#             setupUserInteractionListener();
#             monkeyPatchHistoryMethod('pushState');
#             monkeyPatchHistoryMethod('replaceState');

#             window.addEventListener('load', function() {
#                 setTimeout(() => {
#                     window.__userInteractionInjected__ = false;
#                     setupUserInteractionListener();
#                 }, 300);
#             });

#             const observer = new MutationObserver(() => {
#                 if (window.__lastInjectedBody__ !== document.body) {
#                     window.__userInteractionInjected__ = false;
#                     setupUserInteractionListener();
#                 }
#             });
#             observer.observe(document.documentElement, { childList: true, subtree: true });
#         })();
#     """)

def safe_inject(driver_task):
    while True:
        try:
            # 等待頁面載入完成
            WebDriverWait(driver_task, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            injected = driver_task.execute_script("return window.__userInteractionInjected__ === true;")
            if not injected:
                inject_script(driver_task)
            break
        except Exception as e:
            logging.warning(f"Failed to inject script: {e}")
            time.sleep(2.5)

def get_user_interactions(driver_task):
    """ 從 Selenium 瀏覽器取得使用者互動紀錄 """
    try:
        interactions = driver_task.execute_script("return window.userInteractions || [];")
    except Exception as e:
        logging.warning(f"Failed to get user interactions: {e}")
        return []

    return interactions

def clear_userInteractions(driver_task):
    try:
        script = """
        window.userInteractions = [];
        sessionStorage.setItem('userInteractions', JSON.stringify(window.userInteractions));
        localStorage.setItem('navigationClicks', JSON.stringify([]));
        """
        driver_task.execute_script(script)
    except Exception as e:
        logging.warning(f"Failed to reset user interactions: {e}")

def inject_ok_prompt(driver):
    """
    注入 JavaScript 腳本，為整個網頁介面添加綠色外框，並添加四個覆蓋層用於檢測邊框點擊。
    外框和覆蓋層都使用固定定位，確保它們不會隨著頁面滾動而移動。
    
    Args:
        driver: Selenium WebDriver 實例
    """
    js_script = """
    (function() {
        // 移除現有的邊框和覆蓋層
        document.body.style.border = '';
        ['top', 'bottom', 'left', 'right'].forEach(function(side) {
            var existingOverlay = document.getElementById('ok_prompt_overlay_' + side);
            if (existingOverlay) {
                existingOverlay.remove();
            }
        });

        // 移除舊的邊框容器（如果存在）
        var existingBorderContainer = document.getElementById('ok_prompt_border_container');
        if (existingBorderContainer) {
            existingBorderContainer.remove();
        }

        // 創建一個固定定位的邊框容器
        var borderContainer = document.createElement('div');
        borderContainer.id = 'ok_prompt_border_container';
        borderContainer.style.position = 'fixed';
        borderContainer.style.top = '0';
        borderContainer.style.left = '0';
        borderContainer.style.width = '100%';
        borderContainer.style.height = '100%';
        borderContainer.style.border = '5px solid #28a745';
        borderContainer.style.boxSizing = 'border-box';
        borderContainer.style.pointerEvents = 'none'; // 確保邊框本身不攔截點擊事件
        borderContainer.style.zIndex = '9998'; // 設置層級，低於覆蓋層但高於頁面內容

        // 創建四個覆蓋層，分別覆蓋上、下、左、右邊框
        ['top', 'bottom', 'left', 'right'].forEach(function(side) {
            var overlay = document.createElement('div');
            overlay.id = 'ok_prompt_overlay_' + side;
            overlay.style.position = 'fixed';
            overlay.style.zIndex = '9999';
            overlay.style.backgroundColor = 'transparent';
            overlay.style.cursor = 'pointer';

            if (side === 'top' || side === 'bottom') {
                overlay.style.left = '0';
                overlay.style.width = '100%';
                overlay.style.height = '5px'; // 邊框寬度
                overlay.style[side] = '0';
            } else {
                overlay.style.top = '0';
                overlay.style.height = '100%';
                overlay.style.width = '5px'; // 邊框寬度
                overlay.style[side] = '0';
            }

            // 點擊邊框時設置退出標誌
            overlay.addEventListener('click', function() {
                window.exitInteractionLoop = true;
            });

            document.body.appendChild(overlay);
        });

        // 將邊框容器添加到頁面
        document.body.appendChild(borderContainer);
    })();
    """
    try:
        driver.execute_script(js_script)
    except Exception as e:
        logging.error(f"Failed to inject green border and overlay: {e}")

def userInteraction_to_json_preprocessing(it, interaction, interaction_execution_url, json_recording):
    if interaction['type'] == 'click':
        if 'text' in interaction and interaction['text']:
            clear_record = f"Click on element with text \"{interaction['text']}\""
        else:
            clear_record = f"Click on {interaction['target']} at position ({interaction['x']}, {interaction['y']})"
    
    elif interaction['type'] == 'input':
        clear_record = f"Input \"{interaction['value']}\" into {interaction['target']}"

    elif interaction['type'] == 'scroll':
        clear_record = f"Scroll {interaction['action']} with total_distance: {interaction['total_scroll_distance']}"
    
    elif interaction['type'] == 'navigation':
        clear_record = f"Go back to {interaction['url']}"

    elif interaction['type'] == 'change':
        display_value = interaction.get('selectedText', interaction.get('value', ''))
        clear_record = f"Select \"{display_value}\" in {interaction['target']}"
    
    else:
        clear_record = "unknown interaction"

    json_recording["userInteraction_recording"].append({
        "Interaction_Step": it,
        "Actual_Interaction": clear_record,
        "Executed_On_URL": interaction_execution_url
    })

def userInteractions_recording(start_web, json_recording, record_dir):
    screenshot_dir = os.path.join(record_dir, "screenshot_recording")
    os.makedirs(screenshot_dir, exist_ok=True)

    # 瀏覽器設定
    options = driver_config()

    # 開啟 Selenium 瀏覽器並前往任務頁面
    driver_task = webdriver.Chrome(options=options)

    # 設定瀏覽器頁面
    driver_task.maximize_window()
    driver_task.get(start_web)

    # 點擊 body 觸發互動
    try:
        driver_task.find_element(By.TAG_NAME, 'body').click()
    except:
        pass
    # 防止按空白鍵自動捲動頁面
    driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")

    it = 0
    current_url = None
    it_output = True
    typeinput_buffer = None
    typeinput_buffer_url = None
    select_buffer = None
    select_buffer_url = None
    finish_recording = False
    select_active = False

    # 紀錄起始網站
    logging.info(f'Start from website at {start_web}.')
    json_recording["userInteraction_recording"].append({"Interaction_Step": it, "Actual_Interaction": f"Start from the website at {start_web}.", "Executed_On_URL": "None URL"})
    
    # Iteration 互動迴圈
    while True:
        if finish_recording:
            break

        if it_output:
            it += 1

        # 確保仍然停留在目標視窗
        window_handle_task = driver_task.current_window_handle
        driver_task.switch_to.window(window_handle_task)

        # 確保網頁載入完成
        while True:
            ready_state = driver_task.execute_script("return document.readyState")
            if ready_state == "complete":
                inject_ok_prompt(driver_task)
                break
            else:
                time.sleep(0.5)

        # 擷取螢幕截圖
        img_path = os.path.join(screenshot_dir, 'screenshot_{}.png'.format(it))
        driver_task.save_screenshot(img_path)

        # 注入 JavaScript 監聽用戶行為
        inject_script(driver_task)

        # 跳轉頁面重置
        if current_url == None or driver_task.current_url != current_url:
            scrollY_base = 0
            scrollY_buffer = 0
            scroll_buffer_action = None
            scroll_buffer_url = None
            current_url = driver_task.current_url
        
        # 等待用户交互並記錄
        if it_output:
            logging.info(f'UserInteraction: {it}')
        interactions = []
        while not interactions:
            if not select_active:
                safe_inject(driver_task)
            interactions = get_user_interactions(driver_task)

            # 檢查是否完成紀錄要退出迴圈
            try:
                if not select_active:
                    exit_loop = driver_task.execute_script("return window.exitInteractionLoop === true;")
                    if exit_loop:
                        # 檢查還有沒有buffer要輸出
                        if scroll_buffer_action:
                            scroll_action = {'type': 'scroll', 'action': scroll_buffer_action, 'total_scroll_distance': (scrollY_buffer - scrollY_base)}
                            logging.info(f"User interaction: {scroll_action}")
                            logging.info(f"Interaction is executed on {scroll_buffer_url}")
                            userInteraction_to_json_preprocessing(it=it, interaction=scroll_action, interaction_execution_url=scroll_buffer_url, json_recording=json_recording)
                            scrollY_base = scrollY_buffer
                            scroll_buffer_action = None
                            
                            # for new interaction
                            it += 1
                            logging.info(f'UserInteraction: {it}')
                        # 結束
                        logging.info("########## Finish userInteractions recording ##########")
                        json_recording["userInteraction_recording"].append({"Interaction_Step": it, "Actual_Interaction": f"Task Completed", "Executed_On_URL": current_url})
                        driver_task.quit()
                        finish_recording = True
                        break
            except Exception as e:
                pass

            # 重置檢查項
            it_output = False

            # 未跳轉頁面
            type_interaction_count = 0
            scroll_interaction_count = 0
            select_interaction_count = 0
            for interaction in interactions:
                # 輸入紀錄buffer
                if interaction['type'] == 'input':
                    typeinput_buffer = interaction
                    typeinput_buffer_url = driver_task.current_url
                    type_interaction_count += 1
                
                # 滑動紀錄buffer
                elif interaction['type'] == 'scroll' and (scroll_buffer_action==None or (scroll_buffer_action=='down' and interaction['scrollY']>=scrollY_buffer) or (scroll_buffer_action=='up' and interaction['scrollY']<=scrollY_buffer)):
                    if interaction['scrollY'] > scrollY_base:
                        scroll_buffer_action = 'down'
                    elif interaction['scrollY'] < scrollY_base:
                        scroll_buffer_action = 'up'
                    
                    scrollY_buffer = interaction['scrollY']
                    scroll_buffer_url = driver_task.current_url
                    scroll_interaction_count += 1

                # 滑動方向變換輸出
                elif interaction['type'] == 'scroll' and ((scroll_buffer_action=='down' and interaction['scrollY']<scrollY_buffer) or (scroll_buffer_action=='up' and interaction['scrollY']>scrollY_buffer)):
                    # buffer輸出
                    if typeinput_buffer:
                        logging.info(f"User interaction: {typeinput_buffer}")
                        logging.info(f"Interaction is executed on {typeinput_buffer_url}")
                        userInteraction_to_json_preprocessing(it=it, interaction=typeinput_buffer, interaction_execution_url=typeinput_buffer_url, json_recording=json_recording)
                        typeinput_buffer = None
                        
                        # for new interaction
                        it += 1
                        logging.info(f'UserInteraction: {it}')
                    
                    scroll_action = {'type': 'scroll', 'action': scroll_buffer_action, 'total_scroll_distance': (scrollY_buffer - scrollY_base)}
                    logging.info(f"User interaction: {scroll_action}")
                    logging.info(f"Interaction is executed on {scroll_buffer_url}")
                    userInteraction_to_json_preprocessing(it=it, interaction=scroll_action, interaction_execution_url=scroll_buffer_url, json_recording=json_recording)

                    scrollY_base = scrollY_buffer
                    scroll_buffer_action = None
                    it_output = True

                # 其他輸出
                else:
                    if interaction['type'] in ['click', 'navigation', 'change']:
                        # buffer輸出
                        if typeinput_buffer:
                            logging.info(f"User interaction: {typeinput_buffer}")
                            logging.info(f"Interaction is executed on {typeinput_buffer_url}")
                            userInteraction_to_json_preprocessing(it=it, interaction=typeinput_buffer, interaction_execution_url=typeinput_buffer_url, json_recording=json_recording)
                            typeinput_buffer = None
                            
                            # for new interaction
                            it += 1
                            logging.info(f'UserInteraction: {it}')
                        
                        if scroll_buffer_action:
                            scroll_action = {'type': 'scroll', 'action': scroll_buffer_action, 'total_scroll_distance': (scrollY_buffer - scrollY_base)}
                            logging.info(f"User interaction: {scroll_action}")
                            logging.info(f"Interaction is executed on {scroll_buffer_url}")
                            userInteraction_to_json_preprocessing(it=it, interaction=scroll_action, interaction_execution_url=scroll_buffer_url, json_recording=json_recording)
                            scrollY_base = scrollY_buffer
                            scroll_buffer_action = None
                            
                            # for new interaction
                            it += 1
                            logging.info(f'UserInteraction: {it}')
                        
                        # 下拉選單buffer
                        if interaction['target'] == 'SELECT' and interaction['type'] == 'click':
                            select_active = True
                            select_buffer = interaction
                            select_buffer_url = driver_task.current_url
                            select_interaction_count += 1

                        elif interaction['type'] == 'change' and select_buffer:
                            logging.info(f"User interaction: {select_buffer}")
                            logging.info(f"Interaction is executed on {select_buffer_url}")
                            userInteraction_to_json_preprocessing(it=it, interaction=select_buffer, interaction_execution_url=select_buffer_url, json_recording=json_recording)
                            select_buffer = None
                            select_active = False
                            
                            if scroll_buffer_action:
                                scroll_action = {'type': 'scroll', 'action': scroll_buffer_action, 'total_scroll_distance': (scrollY_buffer - scrollY_base)}
                                logging.info(f"User interaction: {scroll_action}")
                                logging.info(f"Interaction is executed on {scroll_buffer_url}")
                                userInteraction_to_json_preprocessing(it=it, interaction=scroll_action, interaction_execution_url=scroll_buffer_url, json_recording=json_recording)
                                scrollY_base = scrollY_buffer
                                scroll_buffer_action = None
                                
                                # for new interaction
                                it += 1
                                logging.info(f'UserInteraction: {it}')
 
                            # for new interaction
                            it += 1
                            logging.info(f'UserInteraction: {it}')
                        
                        else:
                            select_buffer = None
                            select_active = False
                    
                    if not select_active:
                        logging.info(f"User interaction: {interaction}")
                        logging.info(f"Interaction is executed on {current_url}")
                        userInteraction_to_json_preprocessing(it=it, interaction=interaction, interaction_execution_url=current_url, json_recording=json_recording)
                        it_output = True
            
            # 清空userInteractions
            if len(interactions) != 0:
                clear_userInteractions(driver_task)
            
            # 打字或滑動buffer處理
            if (not it_output) and ((type_interaction_count != 0 and type_interaction_count == len(interactions)) or (scroll_interaction_count != 0 and scroll_interaction_count == len(interactions)) or (select_interaction_count != 0 and select_interaction_count == len(interactions))):
                img_path = os.path.join(screenshot_dir, 'screenshot_{}.png'.format(it+1))
                if not (select_interaction_count != 0 and select_interaction_count == len(interactions)):
                    driver_task.save_screenshot(img_path)
                interactions = []
            
            # 跳轉頁面
            if driver_task.current_url != current_url and not it_output:
                driver_task.execute_script("localStorage.setItem('pendingPopstateInteraction', 'null');")
                safe_inject(driver_task)
                interactions = get_user_interactions(driver_task)
                if interactions:
                    for interaction in interactions:
                        if interaction['type'] in ['click', 'navigation']:
                            # buffer輸出
                            if typeinput_buffer:
                                logging.info(f"User interaction: {typeinput_buffer}")
                                logging.info(f"Interaction is executed on {typeinput_buffer_url}")
                                userInteraction_to_json_preprocessing(it=it, interaction=typeinput_buffer, interaction_execution_url=typeinput_buffer_url, json_recording=json_recording)
                                typeinput_buffer = None
                                
                                # for new interaction
                                it += 1
                                logging.info(f'UserInteraction: {it}')
                            
                            elif scroll_buffer_action:
                                scroll_action = {'type': 'scroll', 'action': scroll_buffer_action, 'total_scroll_distance': (scrollY_buffer - scrollY_base)}
                                logging.info(f"User interaction: {scroll_action}")
                                logging.info(f"Interaction is executed on {scroll_buffer_url}")
                                userInteraction_to_json_preprocessing(it=it, interaction=scroll_action, interaction_execution_url=scroll_buffer_url, json_recording=json_recording)
                                scrollY_base = scrollY_buffer
                                scroll_buffer_action = None
                                
                                # for new interaction
                                it += 1
                                logging.info(f'UserInteraction: {it}')
                            
                            logging.info(f"User interaction: {interaction}")
                            logging.info(f"Interaction is executed on {current_url}")
                            userInteraction_to_json_preprocessing(it=it, interaction=interaction, interaction_execution_url=current_url, json_recording=json_recording)
                            it_output = True
                    # 清空userInteractions
                    clear_userInteractions(driver_task)
                break

    # 最終記錄到json中
    with open(os.path.join(record_dir, "Interactions_recording.json"), 'w', encoding='utf-8') as record_file:
        json.dump(json_recording, record_file, indent=4, ensure_ascii=False)
# --------------------------------------------------------------------

# Agent推斷操作背後的邏輯或原因 ----------------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def clip_message_and_obs(messages:list):
    clipped_messages = []

    # Always keep the system and initial task description
    if len(messages) >= 2:
        clipped_messages.append(messages[0])  # system
        clipped_messages.append(messages[1])  # initial task message

    # Traverse messages in order
    for msg in messages[2:]:
        if msg['role'] == 'assistant':
            # Always keep assistant's replies
            clipped_messages.append(msg)
        elif msg['role'] == 'user':
            # Check if this user message contains images
            if messages.index(msg) == len(messages)-1 or len(msg['content']) == 1:
                clipped_messages.append(msg)
            else:
                # Replace this message to text-only description (no images)
                observation_text = msg['content'][0]['text']
                clipped_msg = {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': f"{observation_text}\n\nObservation screenshots are omitted to save space."},
                    ]
                }
                clipped_messages.append(clipped_msg)
                
    return clipped_messages

def format_msg(it, step_text, before_imgb64, after_imgb64):
    user_prompt = (
        f"Now analyzing action step {it}:\n"
        f"Action Description: \"{step_text}\"\n\n"
        f"The following two screenshots are provided:\n"
    )
    if before_imgb64:
        user_prompt += "1. The FIRST image is the screenshot BEFORE this action.\n"
    else:
        user_prompt += "1. Screenshot BEFORE the action is missing.\n"
    if after_imgb64:
        user_prompt += "2. The LAST image is the screenshot AFTER this action.\n\n"
    else:
        user_prompt += "2. Screenshot AFTER the action is missing.\n\n"

    user_prompt += "Please carefully infer the reasoning and intention behind this step based on the available information."

    curr_msg = {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': f"{user_prompt}"}
        ]
    }
    if before_imgb64:
        curr_msg['content'].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_imgb64}"}})
    if after_imgb64:
        curr_msg['content'].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_imgb64}"}})
    return curr_msg

def call_gemini_api(parameter, gemini_client, messages):
    retry_times = 0
    while retry_times < 10:
        try:
            logging.info('Calling Gemini API...')
            print('Calling Gemini API...')

            contents = []
            for msg in messages:
                parts = []
                role =  msg['role'] if msg['role']!='system' else 'user'
                if isinstance(msg['content'], str):
                    parts.append({"text": msg['content']})
                elif isinstance(msg['content'], list):
                    for item in msg['content']:
                        if item['type'] == 'text':
                            parts.append({"text": item['text']})
                        elif item['type'] == 'image_url':
                            image_url = item['image_url']['url']
                            base64_data = image_url.split("data:image/png;base64,")[1]
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": base64_data
                                }
                            })

                contents.append({
                    "role": role,
                    "parts": parts
                })

            # 呼叫 Gemini API
            gemini_response = gemini_client.generate_content(
                contents=contents,
                generation_config=GenerationConfig(
                    max_output_tokens=1000,
                    temperature=parameter["temperature"]
                )
            )
            return False, gemini_response.text

        except Exception as e:
            logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}')

            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)
            elif type(e).__name__ == 'APIError':
                time.sleep(15)
            elif type(e).__name__ == 'InvalidRequestError':
                return True, None
            else:
                return True, None

        retry_times += 1
        if retry_times == 10:
            logging.info('Retrying too many times')
            return True, None

def ActionReasoning_Agent_thinking(record_dir, parameter):
    logging.info(f'########## ARA start action reasoning ##########')

    # 建立 API 客戶端
    gemini.configure(api_key=parameter["api_key"])
    client = gemini.GenerativeModel(parameter["gemini_model"])

    # 讀取 user interaction 紀錄
    recording_path = os.path.join(record_dir, "Interactions_recording.json")
    with open(recording_path, 'r', encoding='utf-8') as f:
        json_recording = json.load(f)
    steps = json_recording['userInteraction_recording']

    # 找到截圖資料夾
    screenshot_dir = os.path.join(record_dir, "screenshot_recording")

    # 初始化對話記錄與提示詞
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    init_message = {
        'role': 'user',
        'content': f"The Task Goal is: \"{json_recording['task_question']}\".\n"
                   f"You will receive each operation step one by one for reasoning how each operation to finally finish this task.\n"
                   f"Please infer the reasoning behind each step based on the action description and the before/after screenshots."
    }
    messages.append(init_message)

    # 開始逐步處理
    i = 0
    pattern = r'Reasoning:'
    while True:
        step_info = steps[i]
        step_text = step_info['Actual_Interaction']
        current_img_path = os.path.join(screenshot_dir, f"screenshot_{i}.png")
        next_img_path = os.path.join(screenshot_dir, f"screenshot_{i+1}.png")

        # 讀取並轉成 base64
        img_b64_current = encode_image(current_img_path) if os.path.exists(current_img_path) else None
        img_b64_next = encode_image(next_img_path) if os.path.exists(next_img_path) else None

        # 把這次的 user 提示加到 messages
        messages.append(format_msg(i, step_text, img_b64_current, img_b64_next))

        # 控制 messages 長度
        messages = clip_message_and_obs(messages)

        # 呼叫 Gemini API
        error, response_text = call_gemini_api(parameter, client, messages)

        if error:
            logging.error(f"Failed to reason step {i}")
            continue

        # 把 assistant 回答也加入 messages
        reasoning = re.split(pattern, response_text)[1].strip()
        messages.append({
            'role': 'assistant',
            'content': response_text
        })

        json_recording['userInteraction_recording'][i]["Reasoning_of_Executing"] = reasoning
        logging.info(f"Step_{i} reasoning: {reasoning}")
        logging.info(f"Finished reasoning step {i}")
        print(f"Finished reasoning step {i}")
        if i == (len(steps) - 1):
            break
        i += 1
    
    with open(os.path.join(record_dir, "Interactions_recording_with_reason.json"), 'w', encoding='utf-8') as record_file:
        json.dump(json_recording, record_file, indent=4, ensure_ascii=False)
    # 最終記錄到json中
    with open(os.path.join(record_dir, "Interactions_recording_with_reason.json"), 'w', encoding='utf-8') as record_file:
        json.dump(json_recording, record_file, indent=4, ensure_ascii=False)
# --------------------------------------------------------------------

def main(ARA_parameter, task_ques="Find Kevin Duran'ts bio", start_web="https://www.google.com/"):
    # 任務初始化
    task_dir = os.path.join(parameter["all_manual_userDemo"], 'recording_{}'.format(len(os.listdir(parameter["all_manual_userDemo"]))))
    os.makedirs(task_dir, exist_ok=True)
    setup_logger(task_dir)
    logging.info(f'########## {task_ques} ##########')
    json_recording = {"task_question": task_ques, "userInteraction_recording": []}

    # 記錄使用者操作
    userInteractions_recording(start_web=start_web, json_recording=json_recording, record_dir=task_dir)

    # Agent推斷操作邏輯
    ActionReasoning_Agent_thinking(record_dir=task_dir, parameter=ARA_parameter)

    logging.info(f'########## Finish ##########')
    print('End of process')

if __name__ == '__main__':
    with open("parameter.json", 'r') as file:
        parameter = json.load(file)
    os.makedirs(parameter["all_manual_userDemo"], exist_ok=True)
    tasks = []
    with open(parameter["userDemo_file"], 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    for task in tasks:
        main(parameter, task_ques=task["ques"], start_web=task["web"])
    print("All Finish")
