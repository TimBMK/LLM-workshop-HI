# Setup

This workshop will require a few things: 
1) A properly configured Python Environment
2) Access to OpenAI models
3) Access to Hugging Face models
4) A installation of Ollama

Please note that we will **not** have time to set up Python during the workshop. So make sure you take care of this beforehand.

Also note that, while installing Ollama and getting your OpenAI account ready is fast and can be done after the first day's workshop session, getting granted access to the Hugging Face models we'll use may take a moment. So I suggest you set up your Hugging Face Account and request access to these models as soon as possible!

### Hardware Requirements:
We will not need a super computer for this workshop. However, we will run some (Large) Language Models locally, which requires at least some computing power. I would suggest a decent multi-core CPU and 16gb RAM, but you may get away with less. We will **not** however, need a GPU. Anything we do can run on a CPU, even though a GPU will speed things up in some cases. You will also need some disk space, including up to 10gb for setting up your python environment.

If you are worried that your computer does not meet the hardware requirements for this workshop, you can use [Google Collab](https://colab.research.google.com/) or a different remote compute service. Google Collab only requires need a Google account and hardware it provides (including a small GPU) it provides will be enough for our purposes.

## Python

First of all, we will need a running Python installation. While there are many ways to do this, I would recommend to use Visual Studio Code (VSCode) and Conda.

If you have your own setup, that's perfectly fine. In this case, please make sure to have the correct Pyhton version and the required packages installed, preferably in a fresh environment (see below under Python Environment). If you have no running setup (or would like to try another setup), I suggest you follow these steps.

**If you are using Google Collab**, you do not need to install VSCode, but Conda can be helpful for version control.

### VSCode
VSCode is a code editor, which you can use to edit any programming language. It does, however, have some advantages which make it convenient to use for Python. For one, it does have a pretty graphical interface and some convenience functions such as hotkeys. It also supports Jupyter Notebooks without the need to set them up to run in your browser (which I personally find a bit cumbersome). There's also support for some Jupyter features such as inspecting dataframes. Furthermore, it does have integration for Github Copilot, which can be pretty convenient.

VSCode also lets you easily switch between project folders via the _open folder_ command under file, and let's you conveniently pick a Python environment for each script via the _select kernel_ option.  

_By the way:_ If you have a server for remote compute, you can also connect VSCode with relative ease. You can find a tutorial on how to do this [here](https://www.digitalocean.com/community/tutorials/how-to-use-visual-studio-code-for-remote-development-via-the-remote-ssh-plugin). 

#### Download VSCode
You can download VSCode for free here: [https://code.visualstudio.com/Download]()

#### Set up VSCode
In VSCode, you should install the **Python** and **Jupyter** Extensions. You can do this via the "Extensions" tab on the left. 

#### Copilot
There you can also download **Github Copilot** and its **Chat** extension. Installing these gives you functionality for AI-generated code completions and a Chat window (think ChatGPT, integrated in your coding editor). You can find more info here: [https://github.com/features/copilot](https://github.com/features/copilot)

Note that in order to use it, you will need a [Github](https://github.com/) account (which is generally recommended for this course). And while the starting tier is free, the higher tier is also free for students and teachers in academia! You can sign up for the program here: [https://github.com/education](https://github.com/)

Copilot can be a great help in writing cookie cutter code or simple tasks. Especially if you're new to Python, this can be very convenient, as it can also explain what code does. _However_ note that Copilot is not always right, and easily fails for more complex tasks especially. **Always double check AI generated code, and make sure it does what you think it does!**

### Conda
Conda is, in essence, a package manager for Python. While there is full-blown [Anaconda](https://www.anaconda.com/) framework (including numerous additional programs for data science), I personally prefer the more lightweight [Miniconda](https://docs.anaconda.com/miniconda/). Conda allows you to set up multiple Python environments, each with their own Python version and packages. This is very convenient, as Python packages tend to break code when updated. So Conda allows you to keep dedicated environments for different projects without risking the code to break when, say, your latest project requires a newer version of something.

#### Download Conda
You can download Anaconda or Miniconda here: [https://www.anaconda.com/download/](https://www.anaconda.com/download/)

To access the downloads, you can simply skip the registration process (just click "Skip registration" in the window asking for your email address and you're there).

While it is up to you which one you choose, Miniconda will be enough for our purposes.

### Python Environment
Next, we'll need to set up our Python environment. It needs:
1) Python Version 3.12
2) The list of packages defined [here](https://github.com/TimBMK/LLM-workshop-HI/blob/main/setup/packages.txt)

**_Please make sure you're using the correct Python Version and Package Versions, otherwise you might run into problems during the workshop!_**

If you're using conda, you can easily set this up with the following code. For this, you'll need to open the **Anaconda Prompt on Windows** (preferably as an administrator) or the **Terminal** on Mac/Linux.

There, you can create a new environment with Python 3.12 and the name "nlp_workshop" with the following command:
`conda create -n nlp_workshop python=3.12`

_Note_: You can specify the location of the environment when creating it. This is helpful, e.g. when you have limited space on your partition. See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#specifying-a-location-for-an-environment).

Next, you'll need to activate the environment:
`conda activate nlp_workshop`

Now, we can start installing packages in the environment. We'll start with jupyter notebook:
`conda install notebook`

Finally, we can install our packages directly from the list:
`pip install -r https://raw.githubusercontent.com/TimBMK/LLM-workshop-HI/refs/heads/main/setup/packages.txt`

If you use **Google Collab**, you do not need to set up the conda environment, or need to install notebook. Instead, you can install the required packages directly with 
`!pip install -r https://raw.githubusercontent.com/TimBMK/LLM-workshop-HI/refs/heads/main/setup/packages.txt`
(note the `!` infront of `pip` that lets you access the terminal in Collab)

### CUDA
**_Optional_**: If you have a CUDA-capable device you wish to use when running language models, you will need to install a pytorch version that comes shipped with CUDA. In most cases, you will need a NVIDIA GPU to make use of CUDA (but there might be some hacks for AMD cards under linux I am not aware of). **_You can skip this step if you do not have a NVIDIA GPU._**

**If you have a NVIDIA GPU** you want to use, you can check your supported cuda version with the Terminal command `nvidia-smi` (e.g. in the VSCode Terminal in the lower half of the screen). On the [pytorch page](https://pytorch.org/get-started/locally/), you can find out what versions of CUDA are currently supported. You should install the highest version available for your device. We will not need the torchvision and torchaudio packages, but feel free to install them if you want to work with these data types later. 
To install pytorch with the highest supported CUDA version, run:
`pip install torch --index-url https://download.pytorch.org/whl/cu126`
If you need a lower version, check the [pytorch page](https://pytorch.org/get-started/locally/).
*Note*: The pytorch version with CUDA takes a bit ofdisk scpace around 2.5gb



_Note:_ If you run into trouble installing a package with `pip install`, try the `conda install` command. For certain packages, this resolves installation issues.

**If you use VSCode**, you can pick your newly set up environment via the _select kernel_ button. VSCode will also prompt you to pick an environment when attempting to run a script.



## OpenAI
In order to use open AI's GPT models, you will need an Account on their [website](https://openai.com/). Making an account is free. However, you will also need **some money** in your account to use the API. OpenAI charges API use by tokens, which depend on the task carried out. For our purposes, 5€ will be enough, and should give you plenty of extra tokens to play around with later. 

### Billing
You can add money to your account via their [API page](https://platform.openai.com/). After logging in there, you navigate to _settings_ in the top right corner and from there to _billing_ on the left hand side. There you can add a payment method and add credit to your account. While you can set up a billing plan and have OpenAI charge you automatically when you need more tokens, I would recommend adding a fixed amount. The reason is simple: if you play around with the API later and end up running a very large query by accident, you may end up with quite a large bill!

### API Key
Next, we'll need to obtain an API key, so that we can use our credit for requests to the API. For that, navigate to _API keys_ and create a new secret key. 

**Important**: While you can always delete the key and create a new one, after creating it is the _only time you can see it_, so do not close this window until after you saved the key somewhere!

In order to access the key later without directly exposing it in the code (which is bad for all kinds of reasons, mostly security), we will set up a .env file in our working directory. In the directory you wish to work in, create a new file called ".env". Open this with an editor of your choice (any standard texteditor will do), and add the following line:
`OPEN_API_KEY=`
and paste your API key directly after the `=`.
Then save the file and close it.

**Important**: Do not share or expose this file publicly, as it contains sensitive information (your API key!). For example, this means _do not_ upload it to Github!


## Hugging Face
Hugging Face is a website that hosts many, many language models and has established itself as the main hub for machine learning. From there, you can download models, and upload your own. They have also begun to offer inference endpoints, meaning you can use their API to run (Large) Language Models. If you do not have sufficient hardware, but want to try different models that are potentially better or at least cheaper than OpenAI, this can be a good option. 

Signing up is free, as are most models. However, the authors of some models we'll use - a classification model based on the Comparative Agendas Project - have recently begun to ask some user information as part of their fair use policy. This means that, in order to use these models, you will need to sign up for Hugging Face and request access to these models. I was granted access within a few hours, but I would **do this sooner rather than later** so you can be sure to have access during the workshop.

First, go to [https://huggingface.co/](https://huggingface.co/) and sign up for an account.

Next, visit the following two model pages and sign up for access. (I gave them my university contact info and "research" as use reason, which worked fine)
[https://huggingface.co/poltextlab/xlm-roberta-large-english-media-cap-v3](https://huggingface.co/poltextlab/xlm-roberta-large-english-media-cap-v3)
[https://huggingface.co/poltextlab/xlm-roberta-large-german-party-cap-v3](https://huggingface.co/poltextlab/xlm-roberta-large-english-media-cap-v3)

Feel free to browse the other models by poltextLAB and see if anything else peeks your interest that you might want to request access for, maybe for your own research? 
[https://huggingface.co/poltextlab](https://huggingface.co/poltextlab)

In order to access the model later, we will need to set up an **API key** (same as before). 

To do this, click on your account icon in the top right corner of the hugging face page and click on _acces tokens_ on the left hand side. Click on _create new token_. A token with _read_ configuration is enough for our purposes. Copy the key that is shown.

Open the ".env" file in your working directory with a text editor of your choice. In a new line, add
`HF_TOKEN=`
and paste your API key directly after the `=`.
Then save and close it.

**Important**: Do not share or expose this file publicly, as it contains sensitive information (your API key!). For example, this means _do not_ upload it to Github!
In order to access the model later, we will need to set up an **API key** (same as before). 

To do this, click on your account icon in the top right corner of the hugging face page and click on _acces tokens_ on the left hand side. Click on _create new token_. A token with _read_ configuration is enough for our purposes. Copy the key that is shown.

Open the ".env" file in your working directory with a text editor of your choice. In a new line, add
`HF_TOKEN=`
and paste your API key directly after the `=`.
Then save and close it.

**Important**: Do not share or expose this file publicly, as it contains sensitive information (your API key!). For example, this means _do not_ upload it to Github!

## Ollama
Finally, we will need ollama to run quantized LLMs locally. 

You can download ollama for free here: [https://ollama.com/](https://ollama.com/)

Installation is rather straight forward. For windows, it comes with an installer. 
You can check if your installation was successfull by running
`ollama -v`
in a terminal, for example the one you'll find under the _terminal_ tab in the lower part of VSCode. This gives you the ollama version and tells if you there is a running ollama instance.

After installation the instance should be running automatically. If it does not, you can start it with the Terminal command 
`ollama serve`
You can run this, for example, in the terminal provided by VSCode. Note that it "blocks" the session as it runs continuously, and terminates as soon as you close it. If you run the starter application under windows (or keep it in your autostart), running this should not be necessary.






Conda Environment mit Python Version 3.12

```
conda create -n nlp_workshop python=3.12

conda install ipykernel
conda install ipywidgets

pandas==2.2.3
scikit-learn==1.6.1
seaborn==0.13.2
nltk==3.9.1
gensim==4.3.3
transformers==4.48.3
python-dotenv==1.0.1
langchain==0.3.18
langchain_ollama==0.2.3
langchain_openai==0.3.5
langchain_chroma==0.2.2
langchain_community==0.3.17
utils==1.0.2
langgraph==0.2.72
langchain_huggingface==0.1.2

conda install pytorch

```

API Key in .env
- OpenAI
- HuggingFace

Credits in OpenAI account

Ollama installieren