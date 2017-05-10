---
layout: post
title: DeepFace
use_math: true
---

Today, I'll be reviewing ["DeepFace: Closing the Gap to Human-Level Performance in Face Verification"](http://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Taigman_DeepFace_Closing_the_2014_CVPR_paper.html)[^deepface], a paper by Taigman et al. from Facebook AI Research and Tel Aviv University, published in CVPR 2014.

This is the first highly successful application of deep neural networks to face verification, providing approximately human-level performance in **face verification**, a subtask in facial recognition.

The problem of facial verification is this: given an image of a face, verify that it belongs to a certain person. Of course, a system that can do this can likely do other facial recognition tasks as well. This is the typical pipeline for facial recognition:

- Take in an image
- **Detect** where the face is. This lets us crop the image so that it contains just the face.
- **Align** the face. We want the different parts of the face (eyes, mouth, nose, etc.) to be at roughly the same part of the image; this is generally done through affine transformations (translations and rotations).
- **Represent** the face: have some (usually vector) representation which we can easily use to compare.
- **Classify** using the representation.

Their model innovates in the **align** step by doing explicit 3D modeling of the face, and **represent** step by using a deep neural network.

## Model

### Face Alignment

They first do a multi-step process to align the face so the image is as easy to use by the representation as possible. In particular, we'd want to both normalize the _pose_ as well as take into account the facial expressions. The identity of the person should be invariant to both of these characteristics. Though "3D models have fallen out of favor in recent years... However, since faces are 3D objects, done correctly, we believe that it is the right way." To this end, they use a 3D model of the face to do align the image:

![Alignment]({{ site.baseurl }}/assets/deepface/alignment.png)

- Detect 6 fiducial[^fiducial] points in the image. This is done using a Support Vector Regressor to predict point configurations from a LBP histogram image descriptor[^lbp]. Then use this reference point to crop the image. See (a,b) above.
- Find 67 fiducial points on the result, and construct a 3D mesh from these points, mapping each part of the 2D image to a polygon in the mesh. They do this by manually placing the 67 fiducial points onto a reference 3D shape. See (c,d,e,f) above.
- From the 3D mesh, generate a 2D image using piecewise affine transformations on each part of the image. See (g) above.

The result, i.e. (g) above, is the input to the representation learner.

### Representation

They use a deep neural network to learn a probability distribution over the $K$ different "classes" (where each class corresponds to a person).

![Architecture]({{ site.baseurl }}/assets/deepface/architecture.png)

They make some interesting architecture choices. The details are not extremely relevant, but here's the general overview:

- First "preprocessing"[^interpretation] step
  - The input is the 3D-algined image (152 by 152 pixels)
  - This is passed through a convolutional layer, a max-pooling layer, and a convolutional layer. They claim that several levels of pooling would cause the network to lose precise information about the face.
- Main step
  - They use locally connected layers for the next three layers. This is like a convolutional layer but with each part of the image learning a different filter bank, which makes sense because the face has been normalized and the filters relevant for one part of the face probably is not for another.
  - Then they have two fully connected layers at the end, which is pretty standard.
  - Finally, they have a softmax which outputs a distribution over the $K$ different classes.

The features are normalized to reduce sensitivity to illumination differences.

This is trained in the standard way, with the cross-entropy loss using stochastic gradient descent and backprop, using dropout on the first fully-connected layer during training (they don't observe much overfitting, so there's no need for more dropout). They note that the representation is quite sparse, claiming it is due to the multiple layers of ReLU activations[^relu].

### Metric

They experimented with a few metrics: first, the standard inner product between feature vectors, the weighted $\chi^2$ similarity, and finally an interesting Siamese network setup.

The Siamese network distance is computed by inputting the two input images to two clones of the trained network, taking the absolute value element-wise from the resulting feature vectors, and putting it through a fully connected layer and a logistic unit, which outputs a number between 0 and 1. This distance is trained on the dataset, but only by freezing all layers except the two new ones.

## Experiments

![Results]({{ site.baseurl }}/assets/deepface/results.png)

This is trained on $SFC$, a large dataset gathered from Facebook, and validated on the standard $LFW$ dataset, which has difficult cases due to aging, lighting, and poses. They achieve 97% with a single model and **97.35%** with an ensemble, beating the previous state-of-the-art of 96.33%. This is also validated on the YouTube video-level face verification dataset, where it blows the state-of-the-art out of the water and demonstrates generalization.

Efficiency-wise, a single image takes 0.33 seconds to run (including decoding, detection, alignment, deep model, and classification) on a 2.2GHz Intel CPU.

---

### Footnotes

[^deepface]: Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition._ 2015.

[^fiducial]: Fiducial just means [reference point](https://en.wikipedia.org/wiki/Fiducial_marker).

[^lbp]: Ahonen, Timo, Abdenour Hadid, and Matti Pietikainen. "Face description with local binary patterns: Application to face recognition." _IEEE transactions on pattern analysis and machine intelligence_ 28.12 (2006): 2037-2041.

[^interpretation]: They interpret these layers as preprocessing because they are primarily convolutional and have very few parameters, mainly doing edge detection and other low-level feature extraction.

[^relu]: I don't understand why this is true-- ReLU is pretty standard, and the representation should not be more sparse than "usual".

