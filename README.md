# 2022: A Year Full of Amazing AI papers- A Review 🚀
## A curated list of the latest breakthroughs in AI by release date with a clear video explanation, link to a more in-depth article, and code.

While the world is still recovering, research hasn't slowed its frenetic pace, especially in the field of artificial intelligence. More, many important aspects were highlighted this year, like the ethical aspects, important biases, governance, transparency and much more. Artificial intelligence and our understanding of the human brain and its link to AI are constantly evolving, showing promising applications improving our life's quality in the near future. Still, we ought to be careful with which technology we choose to apply.

>"Science cannot tell us what we ought to do, only what we can do."<br/>- Jean-Paul Sartre, Being and Nothingness

Here's curated list of the latest breakthroughs in AI and Data Science by release date with a clear video explanation, link to a more in-depth article, and code (if applicable). Enjoy the read!

**The complete reference to each paper is listed at the end of this repository.** *Star this repository to stay up to date and stay tuned for next year!* ⭐️

Maintainer: [louisfb01](https://github.com/louisfb01), also active on [YouTube](https://www.youtube.com/@whatsai) and as a [Podcaster](https://open.spotify.com/show/4rKRJXaXlClkDyInjHkxq3) if you want to see/hear more about AI!

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40whats_ai)](https://twitter.com/Whats_AI)


Subscribe to my [newsletter](http://eepurl.com/huGLT5) - The latest updates in AI explained every week.


*Feel free to [message me](https://www.louisbouchard.ai/contact/) any interesting paper I may have missed to add to this repository.*

*Tag me on **Twitter** [@Whats_AI](https://twitter.com/Whats_AI) or **LinkedIn** [@Louis (What's AI) Bouchard](https://www.linkedin.com/in/whats-ai/) if you share the list!* And come chat with us in our [Learn AI Together Discord community](https://www.louisbouchard.ai/learn-ai-together/)!

👀 **If you'd like to support my work**, you can check to [Sponsor](https://github.com/sponsors/louisfb01) this repository or support me on [Patreon](https://www.patreon.com/whatsai).

Or support me by wearing cool [merch](https://whatsai.myshopify.com/)!

 ### Watch a complete 2022 rewind in 8 minutes

[<img src="https://imgur.com/MGt3APx.png" width="512"/>](https://youtu.be/tYRTzWHOQio)

----

## The Full List
- [Resolution-robust Large Mask Inpainting with Fourier Convolutions [1]](#1)
- [Stitch it in Time: GAN-Based Facial Editing of Real Videos [2]](#2)
- [NeROIC: Neural Rendering of Objects from Online Image Collections [3]](#3)
- [SpeechPainter: Text-conditioned Speech Inpainting [4]](#4)
- [Towards real-world blind face restoration with generative facial prior [5]](#5)
- [4D-Net for Learned Multi-Modal Alignment [6]](#6)
- [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding [7]](#7)
- [Hierarchical Text-Conditional Image Generation with CLIP Latents [8]](#8)
- [MyStyle: A Personalized Generative Prior [9]](#9)
- [OPT: Open Pre-trained Transformer Language Models [10]](#10)
- [BlobGAN: Spatially Disentangled Scene Representations [11]](#11)
- [A Generalist Agent [12]](#12)
- [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding [13]](#13)
- [Dalle mini [14]](#14)
- [No Language Left Behind: Scaling Human-Centered Machine Translation [15]](#15)
- [Dual-Shutter Optical Vibration Sensing [16]](#16)
- [Make-a-scene: Scene-based text-to-image generation with human priors [17]](#17)
- [BANMo: Building Animatable 3D Neural Models from Many Casual Videos [18]](#18)
- [High-resolution image synthesis with latent diffusion models [19]](#19)
- [Panoptic Scene Graph Generation [20]](#20)
- [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion [21]](#21)
- [Expanding Language-Image Pretrained Models for General Video Recognition [22]](#22)
- [MAKE-A-VIDEO: TEXT-TO-VIDEO GENERATION WITHOUT TEXT-VIDEO DATA [23]](#23)
- [Robust Speech Recognition via Large-Scale Weak Supervision [24]](#24)
- [DreamFusion: Text-to-3D using 2D Diffusion [25]](#25)
- [Imagic: Text-Based Real Image Editing with Diffusion Models [26]](#26)
- [eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers [27]](#27)
- [InfiniteNature-Zero: Learning Perpetual View Generation of Natural Scenes from Single Images [28]](#28)
- [Galactica: A Large Language Model for Science [29]](#29)
- [Real-time Neural Radiance Talking Portrait Synthesis via Audio-spatial Decomposition [30]](#30)
- [ChatGPT: Optimizing Language Models for Dialogue [31]](#31)
- [Production-Ready Face Re-Aging for Visual Effects [32]](#32)
- [Paper references](#references)

---

## Resolution-robust Large Mask Inpainting with Fourier Convolutions [1]<a name="1"></a>
You’ve most certainly experienced this situation once: You take a great picture with your friend, and someone is photobombing behind you, ruining your future Instagram post. Well, that’s no longer an issue. Either it is a person or a trashcan you forgot to remove before taking your selfie that’s ruining your picture. This AI will just automatically remove the undesired object or person in the image and save your post. It’s just like a professional photoshop designer in your pocket, and with a simple click!

This task of removing part of an image and replacing it with what should appear behind has been tackled by many AI researchers for a long time. It is called image inpainting, and it’s extremely challenging...


* Short Video Explanation:<br/>
[<img src="https://imgur.com/d5ClyqP.png" width="512"/>](https://youtu.be/Ia79AvGzveQ)
* Short read: [This AI Removes Unwanted Objects From your Images!](https://www.louisbouchard.ai/lama/)
* Paper: [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/pdf/2109.07161.pdf)
* [Code](https://github.com/saic-mdal/lama)
* [Colab Demo](https://colab.research.google.com/github/saic-mdal/lama/blob/master/colab/LaMa_inpainting.ipynb)
* [Product using LaMa](https://cleanup.pictures/)


## Stitch it in Time: GAN-Based Facial Editing of Real Videos [2]<a name="2"></a>
You've most certainly seen movies like the recent Captain Marvel or Gemini Man where Samuel L Jackson and Will Smith appeared to look like they were much younger. This requires hundreds if not thousands of hours of work from professionals manually editing the scenes he appeared in.
Instead, you could use a simple AI and do it within a few minutes. Indeed, many techniques allow you to add smiles, make you look younger or older, all automatically using AI-based algorithms. It is called AI-based face manipulations in videos and here's the current state-of-the-art in 2022!


* Short Video Explanation:<br/>
[<img src="https://imgur.com/lvgMjzS.png" width="512"/>](https://youtu.be/mqItu9XoUgk)
* Short read: [AI Facial Editing of Real Videos ! Stitch it in Time Explained](https://www.louisbouchard.ai/stitch-it-in-time/)
* Paper: [Stitch it in Time: GAN-Based Facial Editing of Real Videos](https://arxiv.org/abs/2201.08361)
* [Code](https://github.com/rotemtzaban/STIT)


## NeROIC: Neural Rendering of Objects from Online Image Collections [3]<a name="3"></a>
Neural Rendering. Neural Rendering is the ability to generate a photorealistic model in space just like this one, from pictures of the object, person, or scene of interest. In this case, you’d have a handful of pictures of this sculpture and ask the machine to understand what the object in these pictures should look like in space. You are basically asking a machine to understand physics and shapes out of images. This is quite easy for us since we only know the real world and depths, but it’s a whole other challenge for a machine that only sees pixels.
It’s great that the generated model looks accurate with realistic shapes, but what about how it blends in the new scene? And what if the lighting conditions vary in the pictures taken and the generated model looks different depending on the angle you look at it? This would automatically seem weird and unrealistic to us. These are the challenges Snapchat and the University of Southern California attacked in this new research.


* Short Video Explanation:<br/>
[<img src="https://imgur.com/xTpuwcN.png" width="512"/>](https://youtu.be/88Pl9zD1Z78)
* Short read: [Create Realistic 3D Renderings with AI !](https://www.louisbouchard.ai/neroic/)
* Paper: [NeROIC: Neural Rendering of Objects from Online Image Collections](https://arxiv.org/pdf/2201.02533.pdf)
* [Code](https://github.com/snap-research/NeROIC)


## SpeechPainter: Text-conditioned Speech Inpainting [4]<a name="4"></a>
We’ve seen image inpainting, which aims to remove an undesirable object from a picture. The machine learning-based techniques do not simply remove the objects, but they also understand the picture and fill the missing parts of the image with what the background should look like.
The recent advancements are incredible, just like the results, and this inpainting task can be quite useful for many applications like advertisements or improving your future Instagram post. We also covered an even more challenging task: video inpainting, where the same process is applied to videos to remove objects or people.

The challenge with videos comes with staying consistent from frame to frame without any buggy artifacts. But now, what happens if we correctly remove a person from a movie and the sound is still there, unchanged? Well, we may hear a ghost and ruin all our work.

This is where a task I never covered on my channel comes in: speech inpainting. You heard it right, researchers from Google just published a paper aiming at inpainting speech, and, as we will see, the results are quite impressive. Okay, we might rather hear than see the results, but you get the point. It can correct your grammar, pronunciation or even remove background noise. All things I definitely need to keep working on, or… simply use their new model… Listen to the examples in my video!


* Short Video Explanation:<br/>
[<img src="https://imgur.com/JyQ41Qv.png" width="512"/>](https://youtu.be/zIIc4bRf5Hg)
* Short read: [Speech Inpainting with AI !](https://www.louisbouchard.ai/speech-inpainting-with-ai/)
* Paper: [SpeechPainter: Text-conditioned Speech Inpainting](https://arxiv.org/pdf/2202.07273.pdf)
* [Listen to more examples](https://google-research.github.io/seanet/speechpainter/examples/)


## Towards real-world blind face restoration with generative facial prior [5]<a name="5"></a>
Do you also have old pictures of yourself or close ones that didn’t age well or that you, or your parents, took before we could produce high-quality images? I do, and I felt like those memories were damaged forever. Boy, was I wrong!

This new and completely free AI model can fix most of your old pictures in a split second. It works well even with very low or high-quality inputs, which is typically quite the challenge.

This week’s paper called Towards Real-World Blind Face Restoration with Generative Facial Prior tackles the photo restoration task with outstanding results. What’s even cooler is that you can try it yourself and in your preferred way. They have open-sourced their code, created a demo and online applications for you to try right now. If the results you’ve seen above aren’t convincing enough, just watch the video and let me know what you think in the comments, I know it will blow your mind!


* Short Video Explanation:<br/>
[<img src="https://imgur.com/DxxFRLI.png" width="512"/>](https://youtu.be/nLDVtzcSeqM)
* Short read: [Impressive photo restoration by AI !](https://www.louisbouchard.ai/gfp-gan/)
* Paper: [Towards real-world blind face restoration with generative facial prior](https://arxiv.org/pdf/2101.04061.pdf)
* [Code](https://github.com/TencentARC/GFPGAN)
* [Colab Demo](https://colab.research.google.com/drive/1sVsoBd9AjckIXThgtZhGrHRfFI6UUYOo)
* [Online app](https://huggingface.co/spaces/akhaliq/GFPGAN)


## 4D-Net for Learned Multi-Modal Alignment [6]<a name="6"></a>
How do autonomous vehicles see?

You’ve probably heard of LiDAR sensors or other weird cameras they are using. But how do they work, how can they see the world, and what do they see exactly compared to us? Understanding how they work is essential if we want to put them on the road, primarily if you work in the government or build the next regulations. But also as a client of these services.

We previously covered how Tesla autopilot sees and works, but they are different from conventional autonomous vehicles. Tesla only uses cameras to understand the world, while most of them, like Waymo, use regular cameras and 3D LiDAR sensors. These LiDAR sensors are pretty simple to understand: they won’t produce images like regular cameras but 3D point clouds. LiDAR cameras measure the distance between objects, calculating the pulse laser’s traveling time that they project to the object.

Still, how can we efficiently combine this information and have the vehicle understand it? And what does the vehicle end up seeing? Only points everywhere? Is it enough for driving on our roads? We will look into this with a new research paper by Waymo and Google Research...


* Short Video Explanation:<br/>
[<img src="https://imgur.com/AxGLy7p.png" width="512"/>](https://youtu.be/0nJMnw1Ldks)
* Short read: [Combine Lidar and Cameras for 3D object detection - Waymo](https://www.louisbouchard.ai/waymo-lidar/)
* Paper: [4D-Net for Learned Multi-Modal Alignment](https://openaccess.thecvf.com/content/ICCV2021/papers/Piergiovanni_4D-Net_for_Learned_Multi-Modal_Alignment_ICCV_2021_paper.pdf)


## Instant Neural Graphics Primitives with a Multiresolution Hash Encoding [7]<a name="7"></a>
As if taking a picture wasn’t a challenging enough technological prowess, we are now doing the opposite: modeling the world from pictures. I’ve covered amazing AI-based models that could take images and turn them into high-quality scenes. A challenging task that consists of taking a few images in the 2-dimensional picture world to create how the object or person would look in the real world.

Take a few pictures and instantly have a realistic model to insert into your product. How cool is that?!

The results have dramatically improved upon the first model I covered in 2020, called NeRF. And this improvement isn’t only about the quality of the results. NVIDIA made it even better.

Not only that the quality is comparable, if not better, but it is more than 1'000 times faster with less than two years of research.

* Short Video Explanation:<br/>
[<img src="https://imgur.com/8PilczV.png" width="512"/>](https://youtu.be/UHQZBQOVAIU)
* Short read: [NVIDIA Turns Photos into 3D Scenes in Milliseconds](https://www.louisbouchard.ai/nvidia-photos-into-3d-scenes/)
* Paper: [Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)
* [Code](https://github.com/NVlabs/instant-ngp)


## Hierarchical Text-Conditional Image Generation with CLIP Latents [8]<a name="8"></a>
Last year I shared DALL·E, an amazing model by OpenAI capable of generating images from a text input with incredible results. Now is time for his big brother, DALL·E 2. And you won’t believe the progress in a single year! DALL·E 2 is not only better at generating photorealistic images from text. The results are four times the resolution!

As if it wasn’t already impressive enough, the recent model learned a new skill; image inpainting.

DALL·E could generate images from text inputs.

DALL·E 2 can do it better, but it doesn’t stop there. It can also edit those images and make them look even better! Or simply add a feature you want like some flamingos in the background.

Sounds interesting? Learn more in the video or read more below!

* Short Video Explanation:<br/>
[<img src="https://imgur.com/xZlfsJO.png" width="512"/>](https://youtu.be/rdGVbPI42sA)
* Short read: [OpenAI's new model DALL·E 2 is amazing!](https://www.louisbouchard.ai/openais-new-model-dall-e-2-is-amazing/)
* Paper: [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://cdn.openai.com/papers/dall-e-2.pdf)


## MyStyle: A Personalized Generative Prior [9]<a name="9"></a>
This new model by Google Research and Tel-Aviv University is incredible. You can see it as a very, very powerful deepfake that can do anything. 

Take a hundred pictures of any person and you have its persona encoded to fix, edit or create any realistic picture you want.

This is both amazing and scary if you ask me, especially when you look at the results. Watch the video to see more results and understand how the model works!

* Short Video Explanation:<br/>
[<img src="https://imgur.com/FAhVBzM.png" width="512"/>](https://youtu.be/BNWAEvFfFvQ)
* Short read: [Your Personal Photoshop Expert with AI!](https://www.louisbouchard.ai/mystyle/)
* Paper: [MyStyle: A Personalized Generative Prior](https://arxiv.org/abs/2203.17272)
* [Code (coming soon)](https://mystyle-personalized-prior.github.io/)


> Check out [the What's AI podcast](https://open.spotify.com/show/4rKRJXaXlClkDyInjHkxq3) for more AI content in the form of interviews with experts in the field! An invited AI expert and I will cover specific topics, sub-fields, and roles related to AI to teach and share knowledge from the people who worked hard to gather it.


## OPT: Open Pre-trained Transformer Language Models [10]<a name="10"></a>
We’ve all heard about GPT-3 and have somewhat of a clear idea of its capabilities. You’ve most certainly seen some applications born strictly due to this model, some of which I covered in a previous video about the model. GPT-3 is a model developed by OpenAI that you can access through a paid API but have no access to the model itself.

What makes GPT-3 so strong is both its architecture and size. It has 175 billion parameters. Twice the amount of neurons we have in our brains! This immense network was pretty much trained on the whole internet to understand how we write, exchange, and understand text. This week, Meta has taken a big step forward for the community. They just released a model that is just as powerful, if not more and has completely open-sourced it.

* Short Video Explanation:<br/>
[<img src="https://imgur.com/ZBHHYaQ.png" width="512"/>](https://youtu.be/Ejg0OunCi9U)
* Short read: [Meta's new model OPT is GPT-3's closest competitor! (and is open source)](https://www.louisbouchard.ai/opt-meta/)
* Paper: [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068.pdf)
* [Code](https://github.com/facebookresearch/metaseq)


## BlobGAN: Spatially Disentangled Scene Representations [11]<a name="11"></a>
BlobGAN allows for unreal manipulation of images, made super easily controlling simple blobs. All these small blobs represent an object, and you can move them around or make them bigger, smaller, or even remove them, and it will have the same effect on the object it represents in the image. This is so cool!

As the authors shared in their results, you can even create novel images by duplicating blobs, creating unseen images in the dataset [like a room with two ceiling fans](https://youtu.be/mnEzjpiA_4E)! Correct me if I’m wrong, but I believe it is one of, if not the first, paper to make the modification of images as simple as moving blobs around and allowing for edits that were unseen in the training dataset. 

And you can actually play with this one compared to some companies we all know! They shared their code publicly and a Colab Demo you can try right away. Even more exciting is how BlobGAN works. Learn more in the video!

* Short Video Explanation:<br/>
[<img src="https://imgur.com/9ouN5ta.png" width="512"/>](https://youtu.be/mnEzjpiA_4E)
* Short read: [This is a BIG step for GANs! BlobGAN Explained](https://www.louisbouchard.ai/blobgan/)
* Paper: [BlobGAN: Spatially Disentangled Scene Representations](https://dave.ml/blobgan/)
* [Code](https://github.com/dave-epstein/blobgan)
* [Colab Demo](https://colab.research.google.com/drive/1clvh28Yds5CvKsYYENGLS3iIIrlZK4xO?usp=sharing#scrollTo=0QuVIyVplOKu)


## A Generalist Agent [12]<a name="12"></a>
Gato from DeepMind was just published! It is a single transformer that can play Atari games, caption images, chat with people, control a real robotic arm, and more! Indeed, it is trained once and uses the same weights to achieve all those tasks. And as per Deepmind, this is not only a transformer but also an agent. This is what happens when you mix Transformers with progress on multi-task reinforcement learning agents.

Gato is a multi-modal agent. Meaning that it can create captions for images or answer questions as a chatbot. You’d say that GPT-3 can already do that, but Gato can do more… The multi-modality comes from the fact that Gato can also play Atari games at the human level or even do real-world tasks like controlling robotic arms to move objects precisely. It understands words, images, and even physics...

* Short Video Explanation:<br/>
[<img src="https://imgur.com/rr9VUXn.png" width="512"/>](https://youtu.be/xZKSWNv6Esc)
* Short read: [Deepmind's new model Gato is amazing!](https://www.louisbouchard.ai/deepmind-gato/)
* Paper: [A Generalist Agent](https://storage.googleapis.com/deepmind-media/A%20Generalist%20Agent/Generalist%20Agent.pdf)


## Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding [13]<a name="13"></a>
If you thought Dall-e 2 had great results, wait until you see what this new model from Google Brain can do. 

Dalle-e is amazing but often lacks realism, and this is what the team attacked with this new model called Imagen. 

They share a lot of results on their project page as well as a benchmark, which they introduced for comparing text-to-image models, where they clearly outperform Dall-E 2, and previous image generation approaches. Learn more in the video...

* Short Video Explanation:<br/>
[<img src="https://imgur.com/IpwaSvZ.png" width="512"/>](https://youtu.be/qhtYPhPWCsI)
* Short read: [Google Brain's Answer to Dalle-e 2: Imagen](https://www.louisbouchard.ai/google-brain-imagen/)
* Paper: [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://imagen.research.google/paper.pdf)
* [Project page with results](https://imagen.research.google/)

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40whats_ai)](https://twitter.com/Whats_AI)

## DALL·E Mini [14]<a name="14"></a>
Dalle mini is amazing — and YOU can use it!

I'm sure you've seen pictures like those in your Twitter feed in the past few days.
If you wondered what they were, they are images generated by an AI called DALL·E mini.
If you've never seen those, you need to watch this video because you are missing out.
If you wonder how this is possible, well, you are on the perfect video and will know the answer in less than five minutes.

Dalle mini is a free, open-source AI that produces amazing images from text inputs.

* Short Video Explanation:<br/>
[<img src="https://imgur.com/Ux4ItPo.png" width="512"/>](https://youtu.be/qOxde_JV0vI)
* Short read: [How does dalle-mini work?](https://www.louisbouchard.ai/dalle-mini/)
* [Code](https://github.com/borisdayma/dalle-mini)
* [Huggingface official demo](https://huggingface.co/spaces/dalle-mini/dalle-mini)


## No Language Left Behind: Scaling Human-Centered Machine Translation [15]<a name="15"></a>
Meta AI’s most recent model, called “No Language Left Behind” does exactly that: translates across 200 different languages with state-of-the-art quality.
A single model can handle 200 languages. How incredible is that?

We find it difficult to have great results strictly in English while Meta is tackling 200 different languages with the same model, and some of the most complicated and less represented ones that even google translate struggles with...

* Short Video Explanation:<br/>
[<img src="https://imgur.com/OHV5bTU.png" width="512"/>](https://youtu.be/2G4NeG17Eis)
* Short read: [No Language Left Behind](https://www.louisbouchard.ai/no-language-left-behind/)
* [Code](https://github.com/facebookresearch/fairseq/tree/nllb)
* Paper: [No Language Left Behind](https://ai.facebook.com/research/no-language-left-behind/)


## Dual-Shutter Optical Vibration Sensing [16]<a name="16"></a>
They reconstruct sound using cameras and a laser beam on any vibrating surface, allowing them to isolate music instruments, focus on a specific speaker, remove ambient noises, and many more amazing applications.

* Short Video Explanation:<br/>
[<img src="https://imgur.com/kkS7tGw.png" width="512"/>](https://youtu.be/n1M8ZVspJcs)
* Short read: [CVPR 2022 Best Paper Honorable Mention: Dual-Shutter Optical Vibration Sensing](https://www.louisbouchard.ai/cvpr-2022-best-paper/)
* [Project page](https://imaging.cs.cmu.edu/vibration/)
* Paper: [Dual-Shutter Optical Vibration Sensing](https://www.marksheinin.com/_files/ugd/a41a28_7d370603fafd419da387de85d8ecb5b4.pdf?index=true)


## Make-a-scene: Scene-based text-to-image generation with human priors [17]<a name="17"></a>
Make-A-Scene is not “just another Dalle”. The goal of this new model isn’t to allow users to generate random images following text prompt as dalle does — which is really cool — but restricts the user control on the generations.

Instead, Meta wanted to push creative expression forward, merging this text-to-image trend with previous sketch-to-image models, leading to “Make-A-Scene”: a fantastic blend between text and sketch-conditioned image generation.

* Short Video Explanation:<br/>
[<img src="https://imgur.com/bivyUmD.png" width="512"/>](https://youtu.be/K3bZXXjW788)
* Short read: [Produce Amazing Artworks with Text and Sketches!](https://www.louisbouchard.ai/make-a-scene/)
* Paper: [Make-a-scene: Scene-based text-to-image generation with human priors](https://arxiv.org/pdf/2203.13131.pdf)


## BANMo: Building Animatable 3D Neural Models from Many Casual Videos [18]<a name="18"></a>
Create deformable 3D models from pictures with BANMo!

* Short Video Explanation:<br/>
[<img src="https://imgur.com/ulRCcMS.png" width="512"/>](https://youtu.be/jDTy-liFoCQ)
* Short read: [Build Animatable 3D Models with AI](https://www.louisbouchard.ai/banmo/)
* Paper: [BANMo: Building Animatable 3D Neural Models from Many Casual Videos](https://banmo-www.github.io/banmo-cvpr.pdf)
* [Code](https://github.com/facebookresearch/banmo)


## High-resolution image synthesis with latent diffusion models [19]<a name="19"></a>
What do all recent super powerful image models like DALLE, Imagen, or Midjourney have in common? Other than their high computing costs, huge training time, and shared hype, they are all based on the same mechanism: diffusion.
Diffusion models recently achieved state-of-the-art results for most image tasks including text-to-image with DALLE but many other image generation-related tasks too, like image inpainting, style transfer or image super-resolution.

* Short Video Explanation:<br/>
[<img src="https://imgur.com/PanqNAf.png" width="512"/>](https://youtu.be/RGBNdD3Wn-g)
* Short read: [Latent Diffusion Models: The Architecture behind Stable Diffusion](https://www.louisbouchard.ai/latent-diffusion-models/)
* Paper: [High-resolution image synthesis with latent diffusion models](https://arxiv.org/pdf/2112.10752.pdf)
* [Code](https://github.com/CompVis/latent-diffusion)


👀 **If you'd like to support my work**, you can check to [Sponsor](https://github.com/sponsors/louisfb01) this repository or support me on [Patreon](https://www.patreon.com/whatsai).


## Panoptic Scene Graph Generation [20]<a name="20"></a>
Panoptic scene graph generation, or PSG, is a new problem task aiming to generate a more comprehensive graph representation of an image or scene based on panoptic segmentation rather than bounding boxes. It can be used to understand images and generate sentences describing what's happening. This may be the most challenging task for an AI! Learn more below...

* Short Video Explanation:<br/>
[<img src="https://imgur.com/QRQnydw.png" width="512"/>](https://youtu.be/cSsE_H_0Cr8)
* Short read: [One of the Most Challenging Tasks for AI](https://www.louisbouchard.ai/psg/)
* Paper: [Panoptic Scene Graph Generation](https://arxiv.org/abs/2207.11247)
* [Code](https://github.com/Jingkang50/OpenPSG)
* [Dataset](https://psgdataset.org/)


## An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion [21]<a name="21"></a>
Text-to-Image models like DALLE or stable diffusion are really cool and allow us to generate fantastic pictures with a simple text input. But would it be even cooler to give them a picture of you and ask it to turn it into a painting? Imagine being able to send any picture of an object, person, or even your cat, and ask the model to transform it into another style like turning yourself into a cyborg of into your preferred artistic style or adding it to a new scene.

Basically, how cool would it be to have a version of DALLE we can use to photoshop our pictures instead of having random generations? Having a personalized DALLE, while making it much more simple to control the generation as “an image is worth a thousand words”. It would be like having a DALLE model that is just as personalized and addictive as the TikTok algorithm.

Well, this is what researchers from Tel Aviv University and NVIDIA worked on. They developed an approach for conditioning text-to-image models, like stable diffusion I covered last week, with a few images to represent any object or concept through the words you will send along your images. Transforming the object of your input images into whatever you want!

* Short Video Explanation:<br/>
[<img src="https://imgur.com/tpwcGVK.png" width="512"/>](https://youtu.be/f3oXa7_SYek)
* Short read: [Guiding Stable Diffusion with your Images](https://www.louisbouchard.ai/imageworthoneword/)
* Paper: [An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/pdf/2208.01618v1.pdf)
* [Code](https://github.com/rinongal/textual_inversion)


## Expanding Language-Image Pretrained Models for General Video Recognition [22]<a name="22"></a>
We’ve seen AI generate text, then generate images and most recently even generate short videos, even though they still need work. The results are incredible when you think that no one is actually involved in the creation process of these pieces and it only has to be trained once to then be used by thousands of people like stable diffusion is. Still, do these models really understand what they are doing? Do they know what the picture or video they just produced really represents? What does such a model understand when it sees such a picture or, even more complex, a video?

* Short Video Explanation:<br/>
[<img src="https://imgur.com/65Vz6if.png" width="512"/>](https://youtu.be/seb4lmVPEe8)
* Short read: [General Video Recognition with AI](https://www.louisbouchard.ai/general-video-recognition/)
* Paper: [Expanding Language-Image Pretrained Models for General Video Recognition](https://arxiv.org/abs/2208.02816)
* [Code](https://github.com/microsoft/VideoX/tree/master/X-CLIP)


## MAKE-A-VIDEO: TEXT-TO-VIDEO GENERATION WITHOUT TEXT-VIDEO DATA [23]<a name="23"></a>
Meta AI’s new model make-a-video is out and in a single sentence: it generates videos from text. It’s not only able to generate videos, but it’s also the new state-of-the-art method, producing higher quality and more coherent videos than ever before!

* Short Video Explanation:<br/>
[<img src="https://imgur.com/ZAv9FBH.png" width="512"/>](https://youtu.be/MWwESVyHWto)
* Short read: [Make-a-video: The AI Film Maker!](https://www.louisbouchard.ai/make-a-video/)
* Paper: [MAKE-A-VIDEO: TEXT-TO-VIDEO GENERATION WITHOUT TEXT-VIDEO DATA](https://makeavideo.studio/Make-A-Video.pdf)
* [Code](https://github.com/lucidrains/make-a-video-pytorch)


## Robust Speech Recognition via Large-Scale Weak Supervision [24]<a name="24"></a>
Have you ever dreamed of a good transcription tool that would accurately understand what you say and write it down? Not like the automatic YouTube translation tools… I mean, they are good but far from perfect. Just try it out and turn the feature on for the video, and you’ll see what I’m talking about.

Luckily, OpenAI just released and open-sourced a pretty powerful AI model just for that: Whisper.

It understands stuff I can’t even comprehend, not being a native English speaker (listen in the video) and it works for language translation too!

* Short Video Explanation:<br/>
[<img src="https://imgur.com/mPbvHfl.png" width="512"/>](https://youtu.be/uFOkMme19Zs)
* Short read: [OpenAI's Most Recent Model: Whisper (explained)](https://www.louisbouchard.ai/whisper/)
* Paper: [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)
* [Code](https://github.com/openai/whisper)


## DreamFusion: Text-to-3D using 2D Diffusion [25]<a name="25"></a>
We’ve seen models able to take a sentence and generate images. Then, other approaches to manipulate the generated images by learning specific concepts like an object or particular style.

Last week Meta published the Make-A-Video model that I covered, which allows you to generate a short video also from a text sentence. The results aren’t perfect yet, but the progress we’ve made in the field since last year is just incredible.

This week we make another step forward.

Here’s DreamFusion, a new Google Research model that can understand a sentence enough to generate a 3D model of it. You can see this as a DALLE or Stable Diffusion but in 3D.

* Short Video Explanation:<br/>
[<img src="https://imgur.com/kgvlHXu.png" width="512"/>](https://youtu.be/epuU0VRIcjE)
* Short read: [3D Models from Text! DreamFusion Explained](https://www.louisbouchard.ai/dreamfusion/)
* Paper: [DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988)


## Imagic: Text-Based Real Image Editing with Diffusion Models [26]<a name="26"></a>
If you think the recent image generation models like DALLE or Stable Diffusion are cool, you just won’t believe how incredible this one is.
"This one" is Imagic. Imagic takes such a diffusion-based model able to take text and generate images out of it and adapts the model to edit the images. You can generate an image and then teach the model to edit it any way you want.

* Short Video Explanation:<br/>
[<img src="https://imgur.com/rws0FYl.png" width="512"/>](https://youtu.be/gbpPQ5kVJhM)
* Short read: [AI Image Editing from Text! Imagic Explained](https://www.louisbouchard.ai/imagic/)
* Paper: [Imagic: Text-Based Real Image Editing with Diffusion Models](https://arxiv.org/abs/2210.09276)
* [Stable Diffusion implementation](https://github.com/justinpinkney/stable-diffusion/blob/main/notebooks/imagic.ipynb)


## eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers [27]<a name="27"></a>
eDiffi, NVIDIA's most recent model, generates better-looking and more accurate images than all previous approaches like DALLE 2 or Stable Diffusion. eDiffi better understands the text you send and is more customizable, adding a feature we saw in a previous paper from NVIDIA: the painter tool.

* Short Video Explanation:<br/>
[<img src="https://imgur.com/e0g0rNe.png" width="512"/>](https://youtu.be/grwp-ht_ixo)
* Short read: [eDiffi explained: New SOTA Image Synthesis model!](https://www.louisbouchard.ai/ediffi/)
* Paper: [eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](https://arxiv.org/abs/2211.01324)


👀 **If you'd like to support my work**, you can check to [Sponsor](https://github.com/sponsors/louisfb01) this repository or support me on [Patreon](https://www.patreon.com/whatsai).

Or support me by wearing cool [merch](https://whatsai.myshopify.com/)!


## InfiniteNature-Zero: Learning Perpetual View Generation of Natural Scenes from Single Images [28]<a name="28"></a>
Generate infinite new frames as if you would be flying into your image!

* Short Video Explanation:<br/>
[<img src="https://imgur.com/hza3mLh.png" width="512"/>](https://youtu.be/FQzGhukV-l0)
* Short read: [InfiniteNature-Zero: Fly Into Your Pictures With AI!](https://www.louisbouchard.ai/infinitenature-zero/)
* Paper: [InfiniteNature-Zero: Learning Perpetual View Generation of Natural Scenes from Single Images](https://infinite-nature-zero.github.io/static/pdfs/InfiniteNatureZero.pdf)
* [Code](https://github.com/google-research/google-research/tree/master/infinite_nature_zero)


## Galactica: A Large Language Model for Science [29]<a name="29"></a>
Galactica is a large language model with a size comparable to GPT-3, but specialized on scientific knowledge. The model can write whitepapers, reviews, Wikipedia pages, and code. It knows how to cite and how to write equations. It’s kind of a big deal for AI and science.

* Short Video Explanation:<br/>
[<img src="https://imgur.com/HVEKpOY.png" width="512"/>](https://youtu.be/2GfxkCWWzLU)
* Short read: [Galactica: What is it and What Happened?](https://www.louisbouchard.ai/galactica/)
* Paper: [Galactica: A Large Language Model for Science](https://galactica.org/static/paper.pdf)


## Real-time Neural Radiance Talking Portrait Synthesis via Audio-spatial Decomposition [30]<a name="30"></a>
From a single video, they can synthesize the person talking for pretty much any word or sentence in real time with better quality. You can animate a talking head following any audio track in real-time.

* Short Video Explanation:<br/>
[<img src="https://imgur.com/Sk6fDKu.png" width="512"/>](https://youtu.be/JUqnLN6Q4B0)
* Short read: [From Audio to Talking Heads in Real-Time with AI! RAD-NeRF explained](https://www.louisbouchard.ai/rad-nerf/)
* Paper: [Real-time Neural Radiance Talking Portrait Synthesis via Audio-spatial Decomposition](https://arxiv.org/abs/2211.12368)


## ChatGPT: Optimizing Language Models for Dialogue [31]<a name="31"></a>
ChatGPT has taken over Twitter and pretty much the whole internet, thanks to its power and the meme potential it provides. We all know how being able to generate memes is the best way to conquer the internet, and so it worked.

Since you’ve seen numerous examples, you might already know that ChatGPT is an AI recently released to the public by OpenAI, that you can chat with. It is also called a chatbot, meaning you can interact with it conversationally, imitatting a one-on-one human discussion.

What you might not know is what it is and how it works... Watch the video or read the article or blog post below to learn more!

* Short Video Explanation:<br/>
[<img src="https://imgur.com/RpH5S2f.png" width="512"/>](https://youtu.be/AsFgn8vU-tQ)
* Short read: [What is ChatGPT?](https://www.louisbouchard.ai/chatgpt/)
* Blog Post: [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/)


## Production-Ready Face Re-Aging for Visual Effects [32]<a name="32"></a>
Whether it be for fun in a Snapchat filter, for a movie, or even to remove a few wrinkles, we all have a utility in mind for being able to change our age in a picture.

This is usually done by skilled artists using Photoshop or a similar tool to edit your pictures. Worst, in a video, they have to do this kind of manual editing for every frame! Just imagine the amount of work needed for that. Well, here’s a solution and a new problem to this situation... 👇

* Short Video Explanation:<br/>
[<img src="https://imgur.com/QOo0O5N.png" width="512"/>](https://youtu.be/WC03N0NFfwk)
* Short read: [Automatic Re-Aging with AI! Disney’s FRAN Model Explained](https://www.louisbouchard.ai/disney-re-age/)
* Blog Post: [Production-Ready Face Re-Aging for Visual Effects](https://studios.disneyresearch.com/2022/11/30/production-ready-face-re-aging-for-visual-effects/)


---


>If you would like to read more papers and have a broader view, here is another great repository for you covering 2021:
[2021: A Year Full of Amazing AI papers- A Review](https://github.com/louisfb01/best_AI_papers_2021) and feel free to subscribe to my weekly [newsletter](http://eepurl.com/huGLT5) and stay up-to-date with new publications in AI for 2022!


*Tag me on **Twitter** [@Whats_AI](https://twitter.com/Whats_AI) or **LinkedIn** [@Louis (What's AI) Bouchard](https://www.linkedin.com/in/whats-ai/) if you share the list!*

---

## Paper references<a name="references"></a>

[1] Suvorov, R., Logacheva, E., Mashikhin, A., Remizova, A., Ashukha, A., Silvestrov, A., Kong, N., Goka, H., Park, K. and Lempitsky, V., 2022. Resolution-robust Large Mask Inpainting with Fourier Convolutions. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 2149–2159)., https://arxiv.org/pdf/2109.07161.pdf

[2] Tzaban, R., Mokady, R., Gal, R., Bermano, A.H. and Cohen-Or, D., 2022. Stitch it in Time: GAN-Based Facial Editing of Real Videos. https://arxiv.org/abs/2201.08361

[3] Kuang, Z., Olszewski, K., Chai, M., Huang, Z., Achlioptas, P. and Tulyakov, S., 2022. NeROIC: Neural Rendering of Objects from Online Image Collections. https://arxiv.org/pdf/2201.02533.pdf

[4] Borsos, Z., Sharifi, M. and Tagliasacchi, M., 2022. SpeechPainter: Text-conditioned Speech Inpainting. https://arxiv.org/pdf/2202.07273.pdf

[5] Wang, X., Li, Y., Zhang, H. and Shan, Y., 2021. Towards real-world blind face restoration with generative facial prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9168–9178), https://arxiv.org/pdf/2101.04061.pdf

[6] Piergiovanni, A.J., Casser, V., Ryoo, M.S. and Angelova, A., 2021. 4d-net for learned multi-modal alignment. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 15435–15445), https://openaccess.thecvf.com/content/ICCV2021/papers/Piergiovanni_4D-Net_for_Learned_Multi-Modal_Alignment_ICCV_2021_paper.pdf.

[7] Thomas Muller, Alex Evans, Christoph Schied and Alexander Keller, 2022, "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding", https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf

[8] A. Ramesh et al., 2022, "Hierarchical Text-Conditional Image Generation with CLIP Latents", https://cdn.openai.com/papers/dall-e-2.pdf

[9] Nitzan, Y., Aberman, K., He, Q., Liba, O., Yarom, M., Gandelsman, Y., Mosseri, I., Pritch, Y. and Cohen-Or, D., 2022. MyStyle: A Personalized Generative Prior. arXiv preprint arXiv:2203.17272.

[10] Zhang, Susan et al. “OPT: Open Pre-trained Transformer Language Models.” https://arxiv.org/abs/2205.01068

[11] Epstein, D., Park, T., Zhang, R., Shechtman, E. and Efros, A.A., 2022. BlobGAN: Spatially Disentangled Scene Representations. arXiv preprint arXiv:2205.02837.

[12] Reed S. et al., 2022, Deemind: Gato - A generalist agent, https://storage.googleapis.com/deepmind-media/A%20Generalist%20Agent/Generalist%20Agent.pdf

[13] Saharia et al., 2022, Google Brain, Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding, https://gweb-research-imagen.appspot.com/paper.pdf

[14] Dayma, et al., 2021, DALL·E Mini, doi:10.5281/zenodo.5146400

[15] NLLB Team et al., 2022, No Language Left Behind: Scaling Human-Centered Machine Translation

[16] Sheinin, Mark and Chan, Dorian and O’Toole, Matthew and Narasimhan, Srinivasa G., 2022, Dual-Shutter Optical Vibration Sensing, Proc. IEEE CVPR.

[17] Gafni, O., Polyak, A., Ashual, O., Sheynin, S., Parikh, D. and Taigman, Y., 2022. Make-a-scene: Scene-based text-to-image generation with human priors. https://arxiv.org/pdf/2203.13131.pdf

[18] Yang, G., Vo, M., Neverova, N., Ramanan, D., Vedaldi, A. and Joo, H., 2022. Banmo: Building animatable 3d neural models from many casual videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2863-2873).

[19] Rombach, R., Blattmann, A., Lorenz, D., Esser, P. and Ommer, B., 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10684–10695), https://arxiv.org/pdf/2112.10752.pdf

[20] Yang, J., Ang, Y.Z., Guo, Z., Zhou, K., Zhang, W. and Liu, Z., 2022. Panoptic Scene Graph Generation. arXiv preprint arXiv:2207.11247.

[21] Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G. and Cohen-Or, D., 2022. An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion.

[22] Ni, B., Peng, H., Chen, M., Zhang, S., Meng, G., Fu, J., Xiang, S. and Ling, H., 2022. Expanding Language-Image Pretrained Models for General Video Recognition. arXiv preprint arXiv:2208.02816.

[23] Singer et al. (Meta AI), 2022, “MAKE-A-VIDEO: TEXT-TO-VIDEO GENERATION WITHOUT TEXT-VIDEO DATA”, https://makeavideo.studio/Make-A-Video.pdf

[24] Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C. and Sutskever, I., Robust Speech Recognition via Large-Scale Weak Supervision.

[25] Poole, B., Jain, A., Barron, J.T. and Mildenhall, B., 2022. DreamFusion: Text-to-3D using 2D Diffusion. arXiv preprint arXiv:2209.14988.

[26] Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I. and Irani, M., 2022. Imagic: Text-Based Real Image Editing with Diffusion Models. arXiv preprint arXiv:2210.09276.

[27] Balaji, Y. et al., 2022, eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers, https://arxiv.org/abs/2211.01324

[28] Li, Z., Wang, Q., Snavely, N. and Kanazawa, A., 2022. InfiniteNature-Zero: Learning Perpetual View Generation of Natural Scenes from Single Images. In European Conference on Computer Vision (pp. 515–534). Springer, Cham, https://arxiv.org/abs/2207.11148

[29] Taylor et al., 2022: Galactica: A Large Language Model for Science, https://galactica.org/

[30] Tang, J., Wang, K., Zhou, H., Chen, X., He, D., Hu, T., Liu, J., Zeng, G. and Wang, J., 2022. Real-time Neural Radiance Talking Portrait Synthesis via Audio-spatial Decomposition. arXiv preprint arXiv:2211.12368.

[31] OpenAI, 2022: ChatGPT: Optimizing Language Models for Dialogue, https://openai.com/blog/chatgpt/

[32] Loss et al., DisneyResearch, 2022: FRAN, https://studios.disneyresearch.com/2022/11/30/production-ready-face-re-aging-for-visual-effects/
