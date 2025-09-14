# Week 1: The AI Landscape and Generative AI

Welcome to Week 1! This lesson introduces you to the world of artificial intelligence (AI), with a special focus on generative AI. By the end of this week, you’ll understand what generative AI is, how it works, and how it can create text and images.

---

## 1. What is Artificial Intelligence?
Artificial Intelligence (AI) is the science of making computers and machines that can perform tasks that usually require human intelligence. These tasks include understanding language, recognizing images, making decisions, and more.

**Examples of AI in daily life:**
- Voice assistants (like Siri, Alexa)
- Recommendation systems (Netflix, YouTube)
- Self-driving cars

---

## 2. What is Generative AI?
Generative AI is a branch of AI that can create new content—such as text, images, music, or code—rather than just analyzing or acting on existing data.

**Key idea:** Generative AI learns from large amounts of data and then uses that knowledge to generate something new that didn’t exist before.

**Examples:**
- ChatGPT writes essays, stories, or answers questions.
- DALL-E and Stable Diffusion create original images from text prompts.

---

## 3. How Does Generative AI Work?
Generative AI uses special models called neural networks. These models are trained on huge datasets (like books, images, or code) and learn patterns in the data.

**Popular generative models:**
- **Large Language Models (LLMs):** Generate text (e.g., ChatGPT, Gemini, Llama-2)
- **Diffusion Models:** Generate images (e.g., Stable Diffusion, DALL-E)
- **GANs (Generative Adversarial Networks):** Used for images, video, and more

**How training works:**
1. The model sees lots of examples (e.g., millions of sentences or images).
2. It learns the patterns and relationships in the data.
3. When given a prompt, it generates new content based on what it has learned.

---

## 4. Where Does Generative AI Fit?
- **Discriminative Models:** Predict labels or categories (e.g., is this email spam?).
- **Generative Models:** Create new data similar to what they were trained on (e.g., write a poem, generate an image).

---

## 5. Applications and Risks of Generative AI

**Applications:**
- Chatbots (e.g., ChatGPT)
- Code generation (e.g., GitHub Copilot)
- Marketing content, design, and art
- Healthcare reports
- Creating synthetic data for research

**Risks and Challenges:**
- Hallucinations (AI makes things up)
- Bias in outputs
- Copyright and plagiarism
- Misinformation

---

## 6. Conceptual Overview of Generative Models

Generative models learn the probability distribution of data, allowing them to “sample” or create new examples.

**Language Models:**
- Predict the next word (token) given the previous context.
- **Parameters:**
	- **Temperature:** Controls creativity (lower = more predictable, higher = more creative).
	- **Top-p:** Controls diversity of outputs.

**Image Models (Diffusion):**
- Start with random noise and gradually “denoise” it into a meaningful image, guided by text prompts.
- **Parameters:**
	- **Steps:** Number of denoising steps (more = more detailed).
	- **Guidance scale:** Balances creativity and realism.
	- **Seed:** Controls randomness (same seed = same output).

**Simple Math Intuition:**
- **Discriminative:** Learns $P(y|x)$ (probability of label $y$ given input $x$).
- **Generative:** Learns $P(x, y)$ (joint probability), can generate new $x$.

---

## 7. Basics of Text Generation
Text generation means creating new sentences, paragraphs, or even stories using AI.

**How it works:**
- The model predicts the next word in a sentence, one word at a time.
- It uses context to make the text coherent and relevant.

**Example:**
> Prompt: "The future of AI is"
>
> Output: "The future of AI is full of exciting possibilities, with machines helping humans solve complex problems."

**Try it yourself:**
- Online demo: [Hugging Face Text Generation](https://huggingface.co/models?pipeline_tag=text-generation)
- Python code:
```python
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2")
print(generator("The future of AI is", max_length=30)[0]['generated_text'])
```

---

## 8. Basics of Image Generation
Image generation means creating new pictures from scratch using AI.

**How it works:**
- The model learns from millions of images and their descriptions.
- You give it a text prompt (e.g., "a cat riding a bicycle"), and it creates a new image that matches the description.

**Example:**
> Prompt: "A photo of an astronaut riding a horse on Mars"
>
> Output: ![Astronaut riding a horse](https://github.com/CompVis/stable-diffusion/blob/main/assets/stable-samples/txt2img/merged-0006.png?raw=true)

**Try it yourself:**
- Online demo: [Stable Diffusion Demo](https://stablediffusionweb.com/)
- Python code:
```python
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
		"CompVis/stable-diffusion-v1-4", 
		use_auth_token=True
).to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
		image = pipe(prompt)["sample"][0]
		image.save("astronaut_rides_horse.png")
```

---

## 9. Why is Generative AI Important?
- It can help write, design, and create faster than ever before.
- It powers chatbots, creative tools, and even helps scientists discover new drugs.
- It raises important questions about ethics, bias, and the future of work.

---

## 10. Explore More
- [What is Generative AI? (IBM)](https://www.ibm.com/topics/generative-ai)
- [Hugging Face Transformers (text models)](https://github.com/huggingface/transformers)
- [Stable Diffusion (image models)](https://github.com/CompVis/stable-diffusion)
- [YouTube: Generative AI Explained](https://www.youtube.com/results?search_query=generative+ai+explained)

---

**In summary:**
Generative AI is a powerful technology that can create new text and images. This week, you’ve learned the basics of how it works and seen real examples. In the coming weeks, you’ll dive deeper into how these models are built and used in the real world!
