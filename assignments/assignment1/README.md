# Assignment #1: Basic Detection & Recognition

## 1. INTRODUCTION

Your task is to evaluate two popular old-school algorithms: Viola-Jones (VJ) [1] for detection and Local Binary Patterns (LBP) [2]–[4] for recognition on the detected regions. The latter you will implement yourself and compare it to the OpenCV’s/Scikit’s/Matlab’s implementation of LBPs and to the plain pixel-to-pixels comparison.

## 2. CONTENTS
### 1. Viola-Jones

To run a pretrained detector is essentially a three-liner – loading up an image, an XML and calling a prediction). Make sure to load appropriate ear (left/right) cascades. More info at: https://docs.opencv.org/3.4/d2/d99/tutorial js face![](Aspose.Words.847f8688-e3cb-4879-84b0-e5af4b421460.001.png) detection.html.

### 2. Local Binary Patterns

Search for Local Binary Pattern papers on Google Scholar. However, a good start are:

- Multiresolution Gray-Scale and Rotation Invariant Tex- ture Classification with Local Binary Patterns [2] (https: //tinyurl.com/3daky8a3)
- RLBP: Robust Local Binary Pattern [5] (https://tinyurl.com/6wwbntwm)
- A Completed Modeling of Local Binary Pattern Oper- ator for Texture Classification [6] (https://tinyurl.com/ 3be6r6u6)

Of course, you will be using LLMs to help you with the code, but you have to understand every line and make it as simple as possible.

In your experiments you will then compare these manually crafted LBPs to one of the existing implementations available in the library of your chosing.

## 3. Data

Use data available here (500 samples of 100 classes): https: //tinyurl.com/ibb-a1-data, but free to use data of your chosing. Make sure to split the data in the training bit, on which you will be doing all the development and research and the final test split for final scores.

## 3. THINGS YOU NEED TO DO
- Run VJ with different parameters on the images and compute Intersection-over-Union, based on the supplied ground-truths.
- Store all these detected regions as images that you will then use for recognition. Store also cropped regions from ground-truth as you will also run recognition experiments on those to see how good recognition would be if detection would be perfect.
- Run your-LBP, LBP from a library (both with a range of parameters) and plain pixel2pixel (just transform two- dimensional image into one-dimensional vector) to get feature vectors for each image (cropped region). Com- pare the vectors using distance measure of your chosing (cosine, euclidean etc.). Report recognition performance by dividing the number of all correctly classified with the number of all the comparisons made. Identities are stored in identities.txt.
- For your LBP feature extractor there are many options with multiple stages of implementation and add-ons, e.g.: with or without histograms, uniform-version of LBP, different radius levels (the most basic is R=1), different code lengths (the most basic is L=8), possible local region/window overlaps and others.
- Submit the code (without the data) and answer the questions on Eucilnica.ˇ
## 4. GRADING

To help you with the report and programming goals (some are optional extras), check scoring below. Maximum is 35 points.

- up to 8 pts Report on VJ performance using different parameters.
- And then for the cropped images using the best per- forming VJ and the images cropped from ground-truths, meaning 2 × the points below:
- 2 pts × 2 Implementation with report on classifica- tion accuracy of pixel-wise image comparison.
- 2 pts × 2 Implementation with report on clas- sification accuracy of LBP implementation from OpenCV/Scikit/Matlab etc.
- 6 pts × 2 Implementation with report on classifica- tion accuracy of your own basic LBP implementa-

tion.

- Your LBP improvements with classification improve- ment/deterioration reports (each 2 pt, meaning you can go over 35), such as:
- using histograms or not,
- using different radii,
- using different word lengths,
- implementing uniform LBP,
- using different levels of local region overlaps,
- using different input image sizes,
- etc.
## 5. SUBMISSION

Submit the code and answer the required questions on Eucilnica.ˇ You do not need to write a report.

The deadline is November 17, 19:00 (so you have time left for partying or playing games), but feel free to submit till the morning of November 20.

## REFERENCES

1. P. Viola and M. Jones, “Rapid Object Detection Using a Boosted Cascade of Simple Features,” in Computer Society Conference on Computer Vision and Pattern Recognition, vol. 1. IEEE, 2001, pp. I–I.
2. T. Ojala, M. Pietikainen, and T. Maenpaa, “Multiresolution gray-scale and rotation invariant texture classification with local binary patterns,” IEEE Transactions on pattern analysis and machine intelligence, vol. 24, no. 7, pp. 971–987, 2002.
3. T. Ahonen, A. Hadid, and M. Pietikainen,¨ “Face recognition with local binary patterns,” in European conference on computer vision. Springer, 2004, pp. 469–481.
4. C. Silva, T. Bouwmans, and C. Frelicot,´ “An extended center-symmetric local binary pattern for background modeling and subtraction in videos,” in International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, VISAPP 2015, 2015.
5. J. Chen, V. Kellokumpu, G. Zhao, and M. Pietikainen,¨ “Rlbp: Robust local binary pattern.” in BMVC, 2013.
6. Z. Guo, L. Zhang, and D. Zhang, “A completed modeling of local binary pattern operator for texture classification,” IEEE transactions on image processing, vol. 19, no. 6, pp. 1657–1663, 2010.
