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

### Running WebVoyager
After setting up the environment, you can start running WebVoyager. 

 1. Modify the gemini_api_key in `run.sh`

You can run WebVoyager with the following command:
```bash 
bash run.sh
```
