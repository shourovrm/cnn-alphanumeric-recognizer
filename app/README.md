# Handwritten digits recognition

This a **Flask** web app to predict handwritten english capital letters and digits using a deep convolutional neural network built with **Keras** and trained on the **NIST** and **MNIST** dataset.

Live demo : https://alpha-digit-recogn.herokuapp.com/


### Train the model

To re-train a different model or see the existing model, check the [training notebook](AZ_digits.ipynb)

### Test the app locally

(remember to remove the "-cpu" from tensorflow-cpu in requirements.txt)


```
cd app/

pip install -r requirements.txt

python app.py
```

### Deploy the app to heroku

First of all make sure you have a heroku account and heroku CLI installed

Then just use the following commands from the root of the repository

```
heroku login

heroku create <your-app-name-here>
```

Now we are going to push only the app folder to heroku, since the other files are irrelevant for production

```
git init
git commit -am "1st commit"
git subtree push --prefix app/ heroku master
```

If any edit is done, push it again-

```
git add app/
git commit -am "<Commit Message>"
git subtree push --prefix app/ heroku master
```

And That's it ! The app is now running on the cloud and available for anyone to check.

For more additional information about deploying apps to heroku, can check their [guide](https://devcenter.heroku.com/articles/getting-started-with-python)
