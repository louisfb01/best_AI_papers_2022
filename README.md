# 2022: A Year Full of Amazing AI papers- A Review [WORK IN PROGRESS] 📌
## A curated list of the latest breakthroughs in AI by release date with a clear video explanation, link to a more in-depth article, and code.

While the world is still recovering, research hasn't slowed its frenetic pace, especially in the field of artificial intelligence. More, many important aspects were highlighted this year, like the ethical aspects, important biases, governance, transparency and much more. Artificial intelligence and our understanding of the human brain and its link to AI are constantly evolving, showing promising applications improving our life's quality in the near future. Still, we ought to be careful with which technology we choose to apply.

>"Science cannot tell us what we ought to do, only what we can do."<br/>- Jean-Paul Sartre, Being and Nothingness

Here is a work in progress of the most interesting research papers for 2022. In short, it is curated list of the latest breakthroughs in AI and Data Science by release date with a clear video explanation, link to a more in-depth article, and code (if applicable). Enjoy the read!

**The complete reference to each paper is listed at the end of this repository.** *Star this repository to stay up to date!* ⭐️

Maintainer: [louisfb01](https://github.com/louisfb01)

Subscribe to my [newsletter](http://eepurl.com/huGLT5) - The latest updates in AI explained every week.


*Feel free to [message me](https://www.louisbouchard.ai/contact/) any interesting paper I may have missed to add to this repository.*

*Tag me on **Twitter** [@Whats_AI](https://twitter.com/Whats_AI) or **LinkedIn** [@Louis (What's AI) Bouchard](https://www.linkedin.com/in/whats-ai/) if you share the list!*


👀 **If you'd like to support my work** and use W&B (for free) to track your ML experiments and make your work reproducible or collaborate with a team, you can try it out by following [this guide](https://colab.research.google.com/github/louisfb01/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)! Since most of the code here is PyTorch-based, we thought that a [QuickStart guide](https://colab.research.google.com/github/louisfb01/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb) for using W&B on PyTorch would be most interesting to share.

👉Follow [this quick guide](https://colab.research.google.com/github/louisfb01/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb), use the same W&B lines in your code or any of the repos below, and have all your experiments automatically tracked in your w&b account! It doesn't take more than 5 minutes to set up and will change your life as it did for me! [Here's a more advanced guide](https://colab.research.google.com/github/louisfb01/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb) for using Hyperparameter Sweeps if interested :)

🙌 Thank you to [Weights & Biases](https://wandb.ai/) for sponsoring this repository and the work I've been doing, and thanks to any of you using this link and trying W&B!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louisfb01/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)

----

## The Full List
- [Resolution-robust Large Mask Inpainting with Fourier Convolutions [1]](#1)
- [Stitch it in Time: GAN-Based Facial Editing of Real Videos [2]](#2)
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

---


>If you would like to read more papers and have a broader view, here is another great repository for you covering 2021:
[2021: A Year Full of Amazing AI papers- A Review](https://github.com/louisfb01/best_AI_papers_2021) and feel free to subscribe to my weekly [newsletter](http://eepurl.com/huGLT5) and stay up-to-date with new publications in AI for 2022!


*Tag me on **Twitter** [@Whats_AI](https://twitter.com/Whats_AI) or **LinkedIn** [@Louis (What's AI) Bouchard](https://www.linkedin.com/in/whats-ai/) if you share the list!*

---

## Paper references<a name="references"></a>

[1] Suvorov, R., Logacheva, E., Mashikhin, A., Remizova, A., Ashukha, A., Silvestrov, A., Kong, N., Goka, H., Park, K. and Lempitsky, V., 2022. Resolution-robust Large Mask Inpainting with Fourier Convolutions. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 2149–2159)., https://arxiv.org/pdf/2109.07161.pdf

[2] Tzaban, R., Mokady, R., Gal, R., Bermano, A.H. and Cohen-Or, D., 2022. Stitch it in Time: GAN-Based Facial Editing of Real Videos. https://arxiv.org/abs/2201.08361
