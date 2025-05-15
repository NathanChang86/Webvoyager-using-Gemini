import platform
import time
import json
import re
import os
import shutil
import logging

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from RAG_prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY, SYSTEM_PREVIOUS_STEP
import google.generativeai as gemini
from google.generativeai.types import GenerationConfig
from utils import get_web_element_rect, encode_image, extract_information, print_message,\
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, clip_message_and_obs_text_only

import json
from datetime import datetime
from initialize_vactor_db import load_vector_db
from langchain_chroma import Chroma

# RAG
def retrieve_and_generate_steps(user_query: str, web: str, vectorstore: Chroma, parameter: dict, k: int = 3, max_retries: int = 3) -> str:
    try:
        # 配置 Gemini API
        gemini.configure(api_key=parameter["api_key"])
        model = gemini.GenerativeModel(parameter["gemini_model"], generation_config={"temperature": 0.1})

        prompt_template = """
            You are a professional technical document assistant for WebVoyager, a web browsing agent. Your task is to filter relevant information from the provided retrieval results based on the given task goal and compile it into a structured instruction manual with actionable, numbered steps to guide the agent in completing the task.

            ### Task Goal:
            {user_query}

            ### Probably start from website:
            {web}
            This is a reference starting point for the task. The task may involve navigating to other websites as needed.

            ### Retrieval Results:
            {retrieval_results}

            ### Retrieval Results Example:
            Each result contains:
            - section: (The title information)
            - content: (The information retrieved)
            - source: (The source of the information)

           ### Relevance Criteria:
            - The goal is to compile an **instruction manual** that provides actionable steps to achieve the task, preferably starting from or relating to the website ({web}).
            - A result is **relevant** if it:
                - Contains keywords or terminology partially or directly related to the task goal or any possible approach for completing it
                - Includes step-by-step instructions, procedures, or operations that could contribute to task completion
                - Describes key functions, tools, or settings that might be useful for the task (e.g., search bar, filters, or sorting options)
                - Contains configuration details, system behaviors, or technical information that could aid in achieving the goal
                - Provides partial but useful information, even if it only addresses one aspect of the task (e.g., how to use a search interface)
                - Mentions alternative methods or approaches that could accomplish the same goal
                - Describes general platform functionality that can be adapted to the task (e.g., search or navigation features on {web} or related platforms)
                - Relates to the website ({web}) or similar platforms that could support the task
            - A result is **not relevant** if it:
                - Is completely unrelated to the task goal and the reference website ({web}), e.g., a GitHub manual for a task about searching preprints on arxiv.org
                - Contains no keywords or terminology related to any approach for completing the task
                - Provides only general theoretical concepts without practical application
            - Don't use like this:
                - Adapted from general web search and functionality.
                - Your answer must from Retrieval Results. If the retrieval result is not relevant at all to the task goal, then you just output 'No relevant retrieval results found.', don't use other information to create answer.

            ### Filtering Process:
            1. **Check for Valid Retrieval Results**:
            - Evaluate retrieval results to determine if they mention or relate to the task goal or the reference website ({web}), even partially.
            - If the retrieval results are empty (e.g., "No retrieval results provided.") or completely unrelated to the task goal and the reference website (e.g., a manual for a different platform like GitHub when the task involves arxiv.org), return the following fixed response:
            ```
            No relevant retrieval results found.
            ```
            - If the results are partially relevant (e.g., describe general features like search or navigation that can be adapted), proceed to generate steps.
            - Do not attempt to generate steps if no relevant results are available.

            2. **Identify Relevant Information**  
            - Consider whether the retrieved content helps in accomplishing the task through ANY possible approach
            - Even if the information describes just one possible method or only a portion of a method, include it
            - If a section contains even one relevant keyword or concept related to task completion, consider it relevant

            3. **Structured Output**:
            - Organize the relevant information into a step-by-step instruction manual tailored for the task, using the reference website ({web}) as a starting point where applicable.
            - If specific steps are provided in the retrieval results, use them directly.
            - If only general platform features are described, adapt them to the task by generating actionable steps (e.g., for a search task on arxiv.org, include steps like "navigate to arxiv.org, enter the query in the search bar, sort by date").
            - Each step must be actionable, clearly described, and numbered sequentially.
            - Use action-oriented language (e.g., "Click the search button," "Type 'query' into the textbox") to ensure clarity.
            - If multiple methods are available, include them in the same "Steps:" section with clear labels (e.g., "Method 1: Using Search").
            - Ensure step numbers are continuous across all methods (e.g., do not restart numbering for each method).
            - Do NOT use sub-steps (e.g., 1.a, 1.b) or nested lists.
            - For irrelevant results, provide a clear explanation of why they do not contribute to the task goal at the end of the output.

            ### Correspondingly, **Output Format** should STRICTLY follow the format:
            - If the retrieval results are completely unrelated to the task goal and the reference website ({web}), return:
                ```
                No relevant retrieval results found.
                [Why you think no relevant retrieval results are found.]
                ```
            - Otherwise, return a string containing the structured manual with numbered steps. Each step should be concise and actionable. Format as follows:
                ```
                Task Goal: {user_query}
                Steps:
                1. [Actionable step description]
                2. [Actionable step description]
                ...

                source: [The source of the information]
                ```

            ### Instructions for Multiple Methods:
            - Combine relevant information from all retrieval results into a single instruction manual.
            - If multiple sources provide steps for the same method, merge them into a cohesive set of steps.
            - If different sources suggest different methods, present each method under the same "Steps:" section with appropriate labels.
            - If no specific steps are provided but general platform features are described, adapt them to the task, starting from {web} or navigating to other relevant websites as needed.
            - If the task requires navigating to other websites, include steps to transition logically (e.g., "Search for external links on {web}").
                        
            ### Example:
            For a task like "Search for the latest news on climate change":
            ```
            Task Goal: Search for the latest news on climate change
            Steps:
            1. Open your web browser and navigate to www.google.com.
            2. Type 'climate change latest news' into the search bar and press Enter.
            3. Click on a news article from a reputable source like BBC or Reuters.
            ```
              
            Please reason step by step and ensure the manual is structured with clear, actionable steps tailored for a web browsing agent. Ensure there is only one "Steps:" section, and all steps are numbered sequentially without sub-steps. If no relevant retrieval results are found, return the fixed response "No relevant retrieval results found. Make sure the output must follow the **Output Format**, don't have any other thinking_output."
            """

        # 配置檢索器
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

        for attempt in range(max_retries):
            try:
                # 獲取檢索結果
                docs = retriever.invoke(user_query)
                # 格式化檢索結果為字符串
                retrieval_results = "\n".join([
                    f"- Section: {doc.metadata.get('section', 'Unknown')}\n"
                    f"  Content: {doc.page_content}\n"
                    f"  Source: {doc.metadata.get('source', 'Unknown')}"
                    for doc in docs
                ])

                # 構建提示詞
                prompt = prompt_template.format(
                    user_query=user_query,
                    web=web,
                    retrieval_results=retrieval_results or "No retrieval results provided."
                )

                # 使用 Gemini API 生成響應
                response = model.generate_content(prompt)
                steps = response.text.strip()

                # 格式驗證
                if "No relevant retrieval results found." in steps:
                    print(steps)
                    return "No results."
                if not steps.startswith("Task Goal:") or "Steps:" not in steps:
                    print(steps)
                    print("Wrong output format")
                    continue
                
                return steps

            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Failed to generate steps for query: {user_query}"
        return f"Failed to generate steps for query: {user_query}"

    except Exception as e:
        return f"Failed to generate steps due to error: {str(e)}"

# 設置log
def setup_logger(folder_path):
    log_file_path = os.path.join(folder_path, 'agent.log')

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Selenium WebDriver設定
def driver_config(parameter):
    options = webdriver.ChromeOptions()

    options.add_experimental_option(
        "prefs", {
            "download.default_directory": parameter["download_dir"],
            "plugins.always_open_pdf_externally": True
        }
    )
    options.add_argument("disable-blink-features=AutomationControlled")
    return options

# 輸入訊息格式化並生成user prompt(截圖和網站文字)
def format_msg(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text, prev_step_action=""):
    if it == 1:
        init_msg += f"{prev_step_action}\nI've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
        init_msg_format = {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': init_msg},
            ]
        }
        init_msg_format['content'].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{web_img_b64}"}
        })
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"{prev_step_action}\nObservation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': f"{prev_step_action}\nObservation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"},
                    {
                        'type': 'image_url',
                        'image_url': {"url": f"data:image/png;base64,{web_img_b64}"}
                    }
                ]
            }
        return curr_msg

# 輸入訊息格式化並生成user prompt(Accessibility Tree)
def format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree, prev_step_action=""):
    if it == 1:
        init_msg_format = {
            'role': 'user',
            'content': init_msg + '\n' + ac_tree
        }
        return init_msg_format
    else:
        if not pdf_obs:
            curr_msg = {
                'role': 'user',
                'content': f"{prev_step_action}\nObservation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
            }
        else:
            curr_msg = {
                'role': 'user',
                'content': f"{prev_step_action}\nObservation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n{ac_tree}"
            }
        return curr_msg

# 呼叫 gemini API
def call_gemini_api(parameter, gemini_client, messages, img):
    retry_times = 0
    while retry_times < 10:
        try:
            logging.info('Calling Gemini API...')
            #time.sleep(60/parameter["RPM"])

            # 設定請求內容
            # 正確格式化 `messages`
            contents = [{"role": "user", "parts": [{"text": m["content"] if isinstance(m["content"], str) else m["content"][0]["text"]}]} for m in messages]

            # 如果有圖片，加上圖片訊息
            if img:
                contents.append({"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": img}}]})

            # 呼叫 API
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

# 點擊
def exec_action_click(info, web_ele, driver_task):
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(3)

# 輸入
def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    # outer_html = web_ele.get_attribute("outerHTML")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    try:
        # Not always work to delete
        web_ele.clear()
        # Another way to delete
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except:
        pass

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass

    actions.send_keys(type_content)
    actions.pause(2)

    #actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(3)
    return warn_obs

# 滑動
def exec_action_scroll(info, web_eles, driver_task, parameter, obs_info):
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {768*2//3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-768*2//3});")
    else:
        if not parameter["text_only"]:
            scroll_ele_number = int(scroll_ele_number)
            web_ele = web_eles[scroll_ele_number]
        else:
            element_box = obs_info[scroll_ele_number]['union_bound']
            element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
            web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(3)


def main():
    # 參數解析與初始化
    with open("parameter.json", 'r') as file:
        parameter = json.load(file)
    os.environ["GOOGLE_API_KEY"] = parameter["api_key"]
    os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

    # 建立 API 客戶端
    gemini.configure(api_key=parameter["api_key"])
    client = gemini.GenerativeModel(parameter["gemini_model"])

    # 瀏覽器設定
    options = driver_config(parameter)
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36')
    options.add_argument("disable-blink-features=AutomationControlled")

    # 建立輸出資料夾
    # Save Result file
    current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    result_dir = os.path.join(parameter["output_dir"], current_time)
    os.makedirs(result_dir, exist_ok=True)

    # 載入任務資料
    # Load tasks
    tasks = []
    with open(parameter["test_file"], 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))

    # 任務迴圈
    for task_id in range(len(tasks)):
        task = tasks[task_id]
        task_dir = os.path.join(result_dir, 'task{}'.format(task["id"]))
        os.makedirs(task_dir, exist_ok=True)
        setup_logger(task_dir)
        logging.info(f'########## TASK{task["id"]} ##########')

        # 開啟 Selenium 瀏覽器並前往任務頁面
        driver_task = webdriver.Chrome(options=options)

        # About window size, 765 tokens
        # You can resize to height = 512 by yourself (255 tokens, Maybe bad performance)
        driver_task.maximize_window()
        driver_task.get(task['web'])

        # 點擊 body 觸發互動
        try:
            driver_task.find_element(By.TAG_NAME, 'body').click()
        except:
            pass
        # 防止按空白鍵自動捲動頁面
        # sometimes enter SPACE, the page will sroll down
        driver_task.execute_script("""window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        time.sleep(5)

        # 清除下載資料夾 PDF
        # We only deal with PDF file
        for filename in os.listdir(parameter["download_dir"]):
            file_path = os.path.join(parameter["download_dir"], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        download_files = []  # sorted(os.listdir(parameter["download_dir"]))

        fail_obs = ""  # When error execute the action
        pdf_obs = ""  # When download PDF file
        warn_obs = ""  # Type warning
        pattern = r'Thought:|Action:|Observation:'

        # 初始化對話記錄與提示詞
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "
        if parameter["text_only"]:
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT_TEXT_ONLY}]
            obs_prompt = "Observation: please analyze the accessibility tree and give the Thought and Action."

        # 生成 RAG 操作步驟
        if os.path.exists(parameter["manual_db"]):
            vectorstore = load_vector_db(parameter["manual_db"], parameter["api_key"])
            rag_steps = retrieve_and_generate_steps(task['ques'], task['web'], vectorstore, parameter)
        else:
            rag_steps = "No results."
        logging.info(f'RAG Steps:\n{rag_steps}')
        print(rag_steps)

        # 建立初始任務訊息
        today_date = datetime.today().strftime('%Y-%m-%d')
        init_msg = f"""Today is {today_date}. Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. \n"""
        init_msg = init_msg.replace('https://www.example.com', task['web'])
        init_msg += """Before taking action, carefully analyze the contents in [Manuals] below.
        Determine whether [Manuals] contain relevant procedures, constraints, or guidelines that should be followed for this task.
        If so, follow their guidance accordingly. If not, proceed with a logical and complete approach.\n"""

        init_msg += f"""[Manuals]\n"""
        if rag_steps == "No results.":
            init_msg += "No relevant manuals available. Proceed with a logical approach to complete the task.\n"
        else:
            init_msg += f"{rag_steps}\n"
        init_msg = init_msg + obs_prompt

        it = 0
        answer = ""

        # Reflection: Trajectory
        current_history = ""     # Record the steps of the current iteration.

        # Iteration 互動迴圈
        while it < parameter["max_iter"]:
            it += 1
            logging.info(f'Iter: {it}')
            if not fail_obs:
                # 擷取網站畫面或Accessibility Tree
                try:
                    if not parameter["text_only"]:
                        rects, web_eles, web_eles_text = get_web_element_rect(driver_task, fix_color=True)
                    else:
                        accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                        ac_tree, obs_info = get_webarena_accessibility_tree(driver_task, accessibility_tree_path)

                except Exception as e:
                    if not parameter["text_only"]:
                        logging.error('Driver error when adding set-of-mark.')
                    else:
                        logging.error('Driver error when obtaining accessibility tree.')
                    logging.error(e)
                    break

                # 擷取螢幕截圖
                img_path = os.path.join(task_dir, 'screenshot{}.png'.format(it))
                driver_task.save_screenshot(img_path)

                # 編碼圖片 (base64) 給 GPT 用
                # encode image
                b64_img = encode_image(img_path)

                # 組合 Observation 訊息
                # format msg
                if not parameter["text_only"]:
                    curr_msg = format_msg(it, init_msg, pdf_obs, warn_obs, b64_img, web_eles_text, SYSTEM_PREVIOUS_STEP + current_history)
                else:
                    curr_msg = format_msg_text_only(it, init_msg, pdf_obs, warn_obs, ac_tree, SYSTEM_PREVIOUS_STEP + current_history)
                messages.append(curr_msg)
            
            else:
                curr_msg = {
                    'role': 'user',
                    'content': fail_obs
                }
                messages.append(curr_msg)

            # 控制 messages 長度
            # Clip messages, too many attached images may cause confusion
            if not parameter["text_only"]:
                messages = clip_message_and_obs(messages, parameter["max_attached_imgs"])
            else:
                messages = clip_message_and_obs_text_only(messages, parameter["max_attached_imgs"])

            # 呼叫LLM
            LLM_call_error, gemini_response_text = call_gemini_api(parameter, client, messages, b64_img)

            if LLM_call_error:
                break
            else:
                logging.info('API call complete...')

            # 記錄 Gemini 回應內容
            messages.append({'role': 'assistant', 'content': gemini_response_text})

            # remove the rects on the website
            if (not parameter["text_only"]) and rects:
                logging.info(f"Num of interactive elements: {len(rects)}")
                for rect_ele in rects:
                    driver_task.execute_script("arguments[0].remove()", rect_ele)
                rects = []

            # 提取 GPT 提出之動作種類與資訊
            # extract action info
            try:
                assert 'Thought:' in gemini_response_text and 'Action:' in gemini_response_text
            except AssertionError as e:
                logging.error(f"Format ERROR: {e}")
                fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
                continue

            bot_thought = re.split(pattern, gemini_response_text)[1].strip()
            chosen_action = re.split(pattern, gemini_response_text)[2].strip()

            # 簡單記錄網頁狀態
            observation = f"Title: {driver_task.title}"
            trajectory_info = f"Thought {bot_thought}\nAction {chosen_action}\nObservation: {observation}"
            current_history += f"Step {it}:\n"
            current_history += trajectory_info
            current_history += "\n\n"
            current_history += "-----------------------"

            action_key, info = extract_information(chosen_action)
            logging.info(f"Bot Thought: {bot_thought}")
            logging.info(f"Action Key: {action_key}, Action Info: {info}\n")

            fail_obs = ""
            pdf_obs = ""
            warn_obs = ""

            # 根據 GPT 指示執行動作
            # execute action
            try:

                window_handle_task = driver_task.current_window_handle
                driver_task.switch_to.window(window_handle_task)

                if action_key == 'click':
                    if not parameter["text_only"]:
                        click_ele_number = int(info[0])
                        web_ele = web_eles[click_ele_number]
                    else:
                        click_ele_number = info[0]
                        element_box = obs_info[click_ele_number]['union_bound']
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                              element_box[1] + element_box[3] // 2)
                        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                    ele_tag_name = web_ele.tag_name.lower()
                    ele_type = web_ele.get_attribute("type")

                    exec_action_click(info, web_ele, driver_task)

                    # deal with PDF file
                    current_files = sorted(os.listdir(parameter["download_dir"]))
                    if current_files != download_files:
                        # wait for download finish
                        time.sleep(10)
                        current_files = sorted(os.listdir(parameter["download_dir"]))

                        current_download_file = [pdf_file for pdf_file in current_files if pdf_file not in download_files and pdf_file.endswith('.pdf')]
                        if current_download_file:
                            pdf_file = current_download_file[0]
                            pdf_obs = get_pdf_retrieval_ans_from_assistant(client, os.path.join(parameter["download_dir"], pdf_file), task['ques'])
                            shutil.copy(os.path.join(parameter["download_dir"], pdf_file), task_dir)
                            pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                        download_files = current_files

                    if ele_tag_name == 'button' and ele_type == 'submit':
                        time.sleep(10)

                elif action_key == 'wait':
                    time.sleep(5)

                elif action_key == 'type':
                    if not parameter["text_only"]:
                        type_ele_number = int(info['number'])
                        web_ele = web_eles[type_ele_number]
                    else:
                        type_ele_number = info['number']
                        element_box = obs_info[type_ele_number]['union_bound']
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                              element_box[1] + element_box[3] // 2)
                        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                    warn_obs = exec_action_type(info, web_ele, driver_task)
                    if 'wolfram' in task['web']:
                        time.sleep(5)

                elif action_key == 'scroll':
                    if not parameter["text_only"]:
                        exec_action_scroll(info, web_eles, driver_task, parameter, None)
                    else:
                        exec_action_scroll(info, None, driver_task, parameter, obs_info)

                elif action_key == 'goback':
                    driver_task.back()
                    time.sleep(2)

                elif action_key == 'google':
                    driver_task.get('https://www.google.com/')
                    time.sleep(2)

                elif action_key == 'answer':
                    answer = info['content']
                    logging.info(info['content'])
                    logging.info('finish!!')
                    break

                else:
                    raise NotImplementedError
                fail_obs = ""
            except Exception as e:
                logging.error('driver error info:')
                logging.error(e)
                if 'element click intercepted' not in str(e):
                    fail_obs = "The action you have chosen cannot be exected. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
                elif action_key is None:
                    fail_obs = ("Format ERROR: Your Action format is invalid. Please strictly follow the format:\n"
                                "- Click [Numerical_Label]\n"
                                "- Type [Numerical_Label]; [Content]\n"
                                "- Scroll [Numerical_Label or WINDOW]; [up or down]\n"
                                "- Wait\n"
                                "- GoBack\n"
                                "- Google\n"
                                "- ANSWER; [content]\n"
                                "For example: Action: Type [7]; [Kevin Durant]")
                else:
                    fail_obs = ""
                time.sleep(2)

        # 儲存 messages 與成本計算
        #print_message(messages, task_dir)
        driver_task.quit()
        logging.info(f'Finish----------------------')
        if answer != "":
            logging.info(f"task: {task['ques']}\nanswer: {answer}")
        else:
            logging.info(f"task: {task['ques']}\nanswer: Fail task")


if __name__ == '__main__':
    main()
    print('End of process')
