---
layout: single
title: Face Recognition
---

Inspired by the new Iphone-X face unlock, I build my own pipeline and Convolutional Neural Network to detect faces and classify it with probability
![jekyll Image](https://www.mathworks.com/content/mathworks/www/en/discovery/convolutional-neural-network/jcr:content/mainParsys/image_copy.adapt.full.high.jpg/1508999490138.jpg)

## Shell Script for Webcam Photos
For the model training and augmentation, I wanted 100 images of both Brittany and I with a distinctive label in file name to make the segmentation easier down the line for parsing. Looking online I found a software that was a simple brew install called imagesnap. The only parameter it took in was the export path, so a bash script for loop would do the job! I had an if then echo prompt to distinguish between our two photoshoots.

```bash
Echo -n "Take pictures of Nick?"
read answer
if echo "$answer" | grep -iq "^y" ;then
    echo Yes
    for x in {1..100}; do imagesnap images/nick$x.jpg; done
else
    echo No
    for x in {1..100}; do imagesnap images/britt$x.jpg; done
fi
```
## Face Detection & Labeling

With 200 images of each user, it was time to detect a face in each image and once again store in a new file directory. This python script could have been executed at same run time as the bash script, but incase I fat fingered something I couldn't risk asking twice to have Brittany make silly faces to my laptop for a few minutes while I snapchated the phenomenom :P. So with this python script, I'm reading in each image and using a Haar Cascades classifier. It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images (Positive images have faces, negative don't). I used a pre-trained model stored in haarcascade_frontalface_default.xml, more details here at  https://en.wikipedia.org/wiki/Haar-like_featureIt . is then used to detect objects in other images
fjaklsfd
```python
for label in ('nick','britt'):
    counter = 0
    for i in range(1,n):
        try:
            img = cv2.imread('images/raw_images/%s%s.jpg'%(label,i))
            faces = face_cascade.detectMultiScale(img, 1.05, 3,minSize=(200,200))
            (x,y,w,h) = faces[0]
            img = img[y:y+h,x:x+w,:]
            img = cv2.resize(img,(300,300), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite("images/face_images/%s_%s.jpg"%(label,counter),img)
            counter += 1
        except:
            print 'No face detected in frame %s'%i
            pass
```

For the webcam photos, I used a 200 by 200 rectangle for face region, minimum of 3 neighbors and a 1.05 scaleFactor, with these parameters it almost detected a face in every image. Since the first layer of my neural network needs a static 2d array dimensions, I resized every face region captured into a 300x300 array. This is a data transformation very similar to my real-time when I'm capturing a face region and resizing to 300x300 to feed through the network.
## Data Augmentation
For real powerful image classifiction problems (detecting cats, dogs, hotdots or not), more data with labels always makes the network learn better on how to differentiate. I'm not building a very sophisticated model to handle the classification of our faces being in a much different enviroment compared to training. Only 200 images is still a bit small.. the alexnet used X. A popular method of generating more data using previous data is a technique called data augmentation. I tried to get a lot of different face positions/reactions when capturing the intial 100 photos, but I can apply some simple computational photography techniques to flip the photo, add gaussian noise, brighten, darken, etc. Millions of permutations of combinations to do.

![jekyll Image](https://cdn-images-1.medium.com/max/1600/1*bqNylp7FcqIBWg0DrcimUw.png)


In my testing photos I would only expect the flip LR and gaussian blur if the quality was lower. Iteratively I generated more input data by flipping the face horizontally with numpy operation and adding a gaussian blur with a 5x5 kernel. This generated X photos in total and was enough input to start!
```python
for pic in os.listdir('images/face_images'):
    label = pic.split('_')[0]
    img = cv2.imread(os.path.join('images/face_images/',pic))
    for blur in (True,False):
        for flip in (True,False):
            if blur == True and flip == True:
                img = np.fliplr(img)
                img = cv2.GaussianBlur(img, (5,5), 0)
                if label == 'britt':
                    cv2.imwrite('images/train/britt_%s.jpg'%britt_counter,img)
                    britt_counter +=1
                elif label == 'nick':
                    cv2.imwrite('images/train/nick_%s.jpg'%nick_counter,img)
                    nick_counter +=1
            if blur == True and flip == False :
                img = cv2.GaussianBlur(img, (5,5), 0)
                if label == 'britt':
                    cv2.imwrite('images/train/britt_%s.jpg'%britt_counter,img)
                    britt_counter +=1
                elif label == 'nick':
                    cv2.imwrite('images/train/nick_%s.jpg'%nick_counter,img)
                    nick_counter +=1
```

## Model Training

For building my network, I used TFlearn's api for building a cnn in tensorflow. This procedural approach makes building a model very easy. I did not do much hyperparameter tuning as I was getting very great validation accuracy so I stuck with the parameters below. 1 Input layer, 5 hidden layers with ladder scaling nodes from 32 - 128 - 32 using a rectified linear activation function. After every activation layer, I used a max pooling layer with 5. I won't go into details of what these all mean, but it's the bread and butter of why cnn works so great with object classification as it uses a flashlight style approach of scanning images for important features to build weights upon to distinguish between classes. Finally squashing my layers into a final 2 node output layer with a softmax probability of the image containing either Nick or Brittany. Taking the value gives a "probability" value seen in my final artifact.
```python
cnn = input_data(shape=[None, 300, 300, 1], name='input')
cnn = conv_2d(cnn, 32, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 64, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 128, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 64, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = conv_2d(cnn, 32, 5, activation='relu')
cnn = max_pool_2d(cnn, 5)
cnn = fully_connected(cnn, 1024, activation='relu')
cnn = dropout(cnn, 0.8)
cnn = fully_connected(cnn, 2, activation='softmax')
cnn = regression(cnn, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(cnn)
model.load("model.tflearn")
```

I trained my model locally on my macbook pro and saw validation accuracies as seen below within 20 epochs. Notice how this took x minutes running using my CPU.

I just arrived back from AWS re:Invent which is a giant cloud computing conference where I racked up a lot of free AWS credits to see how very powerful GPU instances can train my model. Even though I got $175 in credit over the week... I'm still stingy and chose to use spot instances to get in and out at 1/10th the price. These p3 instances cost $14 an hour, multiply that by 24*365 is over x amount a year! I ran my model on a cheaper g instance and the new released p3 instance. Going from x minutes down to 3 seconds is pretty crazy. The key reason these instances can run much faster is because they have so many cuda cores. Training a neural network like this is just millions of matrix multiplications as the weights propogate the error up and back from the network until the accuracy is at good amount.


## Model Prediction and Result

The biggest computational part of an image recogntion model is just the training. I'm still using the same architecture as above, but now I have the most optimal weights for every node in every layer to multiplicate
![jekyll Image](https://media.giphy.com/media/l1KdbJwcJ748JHymI/giphy.gif)



```python
cap = cv2.VideoCapture('videos/input1.mov')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
out = cv2.VideoWriter('videos/output1.mov',cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), 20, (720,1280))
for i in range(600):
        _, img = cap.read()
        faces = face_cascade.detectMultiScale(img, 1.05, 3,minSize=(200,200))
        (x,y,w,h) = faces[0]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        img_pred = img[y:y+h,x:x+w,:]
        img_pred = cv2.resize(img_pred,(300,300), interpolation = cv2.INTER_CUBIC)
        img_pred = cv2.cvtColor(img_pred,cv2.COLOR_BGR2GRAY)
        pred = np.argmax(model.predict(np.array(img_pred).reshape(-1,300,300,1)))
        preds =  model.predict(np.array(img_pred).reshape(-1,300,300,1))[0]
        nick = round(preds[0],2)
        britt = round(preds[1],2)
        if pred == 0:
            lab = 'Nick : %s'%nick
        elif pred == 1:
            lab = 'Brittany : %s'%britt
        cv2.putText(img,lab, (x-100,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255,3)

```

Welcome to my abode! I’m a current data engineer pursuing my Masters part time
in Computer Science and wanted a place to host all my fun data driven projects. My site will mostly contain projects and implementations in the realm of Machine Learning && Data Science && Cloud Computing. I was studying optimization, statistics, and computational data analysis courses for my undergrad throughout 2012-2017, but now a days I'm more interested in application and scalability of these algos including the use of the { cloud }.


I have used this theme in my own php and ruby blogs.I got inspired by [taylantatli](https://github.com/taylantatli/)'s [moon](https://github.com/taylantatli/moon) and [ramme](https://github.com/taylantatli/ramme) themes to use this theme in Jekyll.

I'm not a designer or something, so I'm sure there is a better way to make this theme. But it's working and looks acceptable for different screen sizes. If something looks extremely ugly and you can't resist to fix it, just send me a PR. I will be grateful.

I see some people using this theme. I need to search on Github to find who use it. But I don't want to search like this. If you like this theme or using it, please give a **star** for motivation.

## Installation
* Fork the [Repo](https://github.com/alperenbozkurt/JBlog/fork)
* Edit `_config.yml` file.
* Remove sample posts from `_posts` folder and add yours.
* Edit `index.md` file in `about` folder.
* Change repo name to `YourUserName.github.io`    

That's all.

## Scaffolding    
How JBlog is organized and what the various files are. All posts, layouts, includes, stylesheets, assets, and whatever else is grouped nicely under the root folder.    

{% highlight text %}
├── face_ar.py
├── face_grab.py
├── face_augmentation.py
├── model_train.py
├── images
│   ├── face_images   
│   │   ├── nick_*.jpg                             
│   │   ├── britt_*.jpg                           
│   ├── raw_images   
│   │   ├── nick_*.jpg                            
│   │   ├── britt_*.jpg                                
│   ├── train   
│   │   ├── nick_*.jpg                            
│   │   ├── britt_*.jpg                              
├── videos                                    
│   ├── input_*.mov   
│   └── navigation.yml                        

{% endhighlight %}   

---

## Site Setup
A quick checklist of the files you’ll want to edit to get up and running.    

### Site Wide Configuration
`_config.yml` is your friend. Open it up and personalize it. Most variables are self explanatory but here's an explanation of each if needed:

#### title

The title of your site... shocker!

Example `title: My Awesome Site`

#### url

Used to generate absolute urls in `sitemap.xml`, `feed.xml`, and for generating canonical URLs in `<head>`. When developing locally either comment this out or use something like `http://localhost:4000` so all assets load properly. *Don't include a trailing `/`*.

Examples:

{% highlight yaml %}
url: http://alperenbozkurt.net/JBlog
url: http://localhost:4000
url: //cooldude.github.io
url:
{% endhighlight %}

#### Google Analytics and Webmaster Tools

Google Analytics UA and Webmaster Tool verification tags can be entered in `_config.yml`. For more information on obtaining these meta tags check [Google Webmaster Tools](http://support.google.com/webmasters/bin/answer.py?hl=en&answer=35179) and [Bing Webmaster Tools](https://ssl.bing.com/webmaster/configure/verify/ownership) support.

---

### Navigation Links

To set what links appear in the top navigation edit `_data/navigation.yml`. Use the following format to set the URL and title for as many links as you'd like. *External links will open in a new window.*

{% highlight yaml %}
- title: Home
  url: /

- title: Blog
  url: /blog/

- title: Projects
  url: /projects/

- title: About
  url: /about/

- title: JBlog
  url: http://alperenbozkurt.net/JBlog
{% endhighlight %}

---

## Layouts and Content

Explanations of the various `_layouts` included with the theme and when to use them.

### Post and Page

These two layouts are almost similar. Only difference is page layout doesn't show date under title.

### Post Index Page

A [sample index page]({{ site.url }}/blog/) listing all blog posts. The name can be customized to your liking by editing a few references. For example, to change **Blog** to **Posts** update the following:

* In `_data/navigation.yml`: rename the title and URL to the following:
{% highlight yaml %}
  - title: Posts
    url: /posts/
{% endhighlight %}
* Rename `blog/index.md` to `posts/index.md` and update the YAML front matter accordingly.

### Thumbnails for OG and Twitter Cards

Site logo is used by [Open Graph](https://developers.facebook.com/docs/opengraph/) and [Twitter Cards](https://dev.twitter.com/docs/cards).

**Pro-Tip**: You need to [apply for Twitter Cards](https://dev.twitter.com/docs/cards) before they will begin showing up when links to your site are shared.
{:.notice}

### Kramdown Table of Contents

To include an auto-generated **table of contents** for posts and pages, add the following `_include` before the actual content. [Kramdown will take care of the rest](http://kramdown.rubyforge.org/converter/html.html#toc) and convert all headlines into list of links.

{% highlight html %}
{% raw %}{% include toc.html %}{% endraw %}
{% endhighlight %}

---

## Questions?

Found a bug or aren't quite sure how something works? By all means [file a GitHub Issue](https://github.com/alperenbozkurt/JBlog/issues/new). And if you make something cool with this theme feel free to let me know.

---

## License

This theme is free and open source software, distributed under the MIT License. So feel free to use this Jekyll theme on your site without linking back to me or including a disclaimer.
