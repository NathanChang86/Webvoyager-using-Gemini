import fitz, json, os, shutil
from typing import List, Dict
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# PDF 處理
def extract_pdf_content(pdf_path: str) -> List[Dict[str, str]]:
    """
    從 PDF 檔案中提取文字和圖片，轉換為結構化格式。
    Args:
        pdf_path: PDF 檔案路徑
    Returns:
        List of dicts, each containing page text and image paths (if any)
    """
    doc = fitz.open(pdf_path)
    content = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # 提取圖片
        images = []
        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = f"../data/pdf_images/page_{page_num}_img_{img_index}.{image_ext}"
            os.makedirs("../data/pdf_images", exist_ok=True)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            images.append(image_path)
        
        content.append({
            "page_num": str(page_num + 1),
            "text": text.strip(),
            "images": images
        })
    
    doc.close()
    return content

# JSON 處理
def extract_json_content(json_path: str) -> List[Dict[str, str]]:
    """
    從 JSON 檔案中提取完整操作流程，轉換為單一結構化格式。
    Args:
        json_path: JSON 檔案路徑
    Returns:
        List of dicts, containing a single dict with full process details
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    content = []
    task_question = data.get("task_question", "")
    full_text = f"Task: {task_question}\n\n"
    for step in data.get("userInteraction_recording", []):
        if step['Actual_Interaction'] != "Task Completed":
            full_text += (
                f"Step {step['Interaction_Step']}\n"
                f"Interaction: {step['Actual_Interaction']}\n"
                f"URL: {step['Executed_On_URL']}\n"
                f"Reasoning: {step.get('Reasoning_of_Executing', '')}\n\n"
            )
        else:
            full_text += (
                f"Step {step['Interaction_Step']}\n"
                f"Interaction: {step['Actual_Interaction']}\n"
                f"URL: {step['Executed_On_URL']}\n"
                f"Reasoning: User think already get the answer\n\n"
            )
    content.append({
        "step_num": "full",
        "text": full_text.strip(),
        "task_question": task_question
    })
    
    return content

# 初始化向量資料庫
def initialize_vector_db(pdf_folder: str, demo_folder: str, api_key: str, vectorstore_dir: str) -> Chroma:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=api_key)
    vectorstore = Chroma(
        collection_name="manual",
        embedding_function=embeddings,
        persist_directory=vectorstore_dir,
    )

    documents = []
    if os.path.exists(pdf_folder):
        for pdf_path in os.listdir(pdf_folder):
            # 提取 PDF 內容
            pdf_content = extract_pdf_content(os.path.join(pdf_folder, pdf_path))
            
            # 創建 Document 物件
            for page in pdf_content:
                content = f"Page {page['page_num']}\n\n{page['text']}"
                metadata = {
                    "source_file": pdf_path,
                    "page_num": page["page_num"],
                    "image_paths": json.dumps(page["images"])
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
    
    if os.path.exists(demo_folder):
        for json_path in os.listdir(demo_folder):
            true_path = os.path.join(os.path.join(demo_folder, json_path), "Interactions_recording_with_reason.json")
            json_content = extract_json_content(true_path)

            for step in json_content:
                content = step["text"]
                metadata = {
                    "source_file": json_path,
                    "step_num": step["step_num"],
                    "task_question": step["task_question"],
                    "type": "json"
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

    # 嵌入向量資料庫
    if documents:
        vectorstore.add_documents(documents)
        print(f"成功嵌入")
    else:
        print("警告：資料夾中未找到任何檔案")
    return vectorstore

# 讀取向量資料庫
def load_vector_db(vectorstore_dir: str, api_key: str) -> Chroma:
    """
    從指定的儲存路徑載入 Chroma 向量資料庫。
    Args:
        vectorstore_dir: 向量資料庫儲存路徑
        api_key: Google API 金鑰
    Returns:
        Chroma 向量資料庫實例
    """
    # 初始化相同的嵌入函數
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=api_key)
    
    # 從持久化路徑載入向量資料庫
    vectorstore = Chroma(
        collection_name="manual",
        embedding_function=embeddings,
        persist_directory=vectorstore_dir
    )
    
    return vectorstore


if __name__ == '__main__':
    # 參數解析與初始化
    with open("parameter.json", 'r') as file:
        parameter = json.load(file)
    os.environ["GOOGLE_API_KEY"] = parameter["api_key"]

    if os.path.exists(parameter["manual_db"]):
        shutil.rmtree(parameter["manual_db"])

    # 初始化向量資料庫
    initialize_vector_db(parameter["all_manual_pdf"], parameter["all_manual_userDemo"], parameter["api_key"], parameter["manual_db"])
    print("Finish")
