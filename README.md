#  nlp API

## Installing using GitHub
#### If you're using a terminal to run the bot, you can clone the project repository using the following command:

```bash
git clone https://github.com/vkleshko/nlp-API
```

#### Create a virtual environment:

```bash
python3 -m venv venv
```
#### Activate the virtual environment:

- Linux, macOS
```bash
source venv/bin/activate
```

- Windows
```bash
venv\Scripts\activate
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

```
python -m uvicorn app.main:app --reload 
```
