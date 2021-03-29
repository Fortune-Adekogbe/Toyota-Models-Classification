# **Toyota Model Recognition Project**
A machine learning project that classifies toyota vehicle models.
A mini project to accept a jpeg image of a toyota vehicle, and output the name of the Toyota model.

## To Run the Server:
Fork and clone the repository.
```
$ git clone git-url
```
Navigate to the project directory
```
$ cd path_to_directory
```
Create a virtual environment
```
$ python -m venv name_of_env
```
Activate virtual environment
```
$ name_of_env\Scripts\activate.bat
```
Install the requirements:
```
$ pip install -r requirements.txt
```
Start the local flask server
```
$ python Toyota-Models-Classification/api/app
```

## To make predictions:
create a python file and in it add the following lines of code
```
import requests

resp = requests.post(
    'http://127.0.0.1:5000/predict',
    files={
        "file": open('Image_Path', 'rb')
    }
)

print(resp.json())
```
Run the python file to see the results in the form:
```
{
  class_id: ...,
  class_name: ...,
}
```
where `class_name` is the name of the model and `class_id` is how the predictor identifies it.

### Note
This API was not deployed to heroku because the slug size exceeds the limit of my account. I will explore torchscript as an alternative but till then, I hope you find this useful.
