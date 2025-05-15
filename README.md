# Webvoyager using gemini_model

## GitHub repository URL
[GitHub repository](https://github.com/NathanChang86/Webvoyager-using-Gemini)

## Project Description 
This project is an Agentic AI system designed to automate web interactions using LLMs like GPT-4 or Gemini. The system follows a reflexive perception-decision-action cycle and supports multimodal input (screenshot + element info).  
This implementation is based on and adapted from the **WebVoyager** framework.
[WebVoyager Paper](https://arxiv.org/abs/2401.13919)


## Setup Environment

We use Selenium to build the online web browsing environment. 
 - Make sure you have installed Chrome. (Using the latest version of Selenium, there is no need to install ChromeDriver.)
 - If you choose to run your code on a Linux server, we recommend installing chromium. (eg, for CentOS: ```yum install chromium-browser```) 
 - Create a conda environment for WebVoyager and install the dependencies.
    ```bash
    conda create -n webvoyager_gemini python=3.10
    conda activate webvoyager_gemini
    pip install -r requirements.txt
    ```

## Running
### setting up the environment
 - Modify the gemini_api_key in `parameter.json`

### User_demo
 - If you want to demo the iteraction, you can use this code by following command:
```
python user_action_recorder.py
```

### Initialize vector database
 - If you add new manuals, like pdf_manuals or user_demo_manuals, you need to run this code to add this munuals into vector_db
 - You can use this code by following command:
```
python initialize_vactor_db.py
```

### Running WebVoyager
You can run WebVoyager with the following command:
```
python run.py
```
