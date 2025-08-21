
# 🧠 Emotion-Aware MLP on MNIST  

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)


* A Human-Inspired Emotion-Aware MLP for MNIST Classification. A human-like twist to learning with just Python & NumPy*  

---

## 🌟 Why build this from scratch?
Most AI models today rely on big libraries like PyTorch or TensorFlow. They’re powerful, but they hide the actual learning process.  

This project rebuilds an MNIST digit classifier **from scratch using only Python and NumPy**.  
That way, we *see AI’s brain at its rawest form*.  

Think of it like cooking 👩‍🍳:  
- Instant food 🍜 → Quick, but soulless.  
- Cooking from scratch 🍲 → Slower, but you *understand the ingredients* and enjoy the process.
- Finding the joy in cooking, feeling the happiness when someone preises your cooking or when your mother cooks you your favorite food with lot of love.

That’s exactly what we did here — we cooked up AI from scratch.  

---

## 🤔 Why another MNIST-from-scratch project?

You might be thinking:
"There are already thousands of MNIST projects in Python + NumPy. Why do we need one more?"

Here’s the twist:
Most projects stop at accuracy. They build the same old recipe — input → hidden layers → output.

But ask yourself this:

Does a student learn only by repeating?

Does a cook follow a recipe blindly?

Does a human brain treat easy and hard problems the same way?

The answer is NO.

Humans don’t just compute.
We feel confusion, adjust strategies, focus harder when needed, and gain confidence.

That’s what this project brings:
👉 An Emotional Gate inside a neural network.

It’s not about just recognizing digits.
It’s about giving AI the power to feel its own uncertainty — and learn better because of it.

---

## 💡 What makes this project unique?
Normal AI does this:  
**Input → Hidden Layers → Output**  

But humans don’t just compute. We also **feel**.  
When your mom cooks, it’s not just following a recipe:  
- She tastes while cooking 😋  
- Adjusts spices 🌶️  
- Serves with confidence and joy ❤️  

That “extra layer of feeling” is what we added — an **Emotional Gate**.  

---

## 🔑 Why add the Emotional Gate *after hidden layers*?
Hidden layers are like the **thinking brain** 🧠, chopping and mixing ingredients (edges, curves, patterns).  

But before serving the dish (final prediction), emotions step in:  
- “Does it taste right?”  
- “Am I confident enough to serve it?”  

So the **Emotional Gate** acts like a **taste test** 🍲 before making the final decision.  

---

## ⚖️ With vs Without Emotional Gate  

**Without Emotional Gate (Normal Neural Net):**  
🤖 Like a robot cook:  
- Follows the recipe exactly.  
- Doesn’t taste or adjust.  
- Food may be edible… but bland.
- The network treats every sample the same.
- Sometimes overfits on easy examples.
- Sometimes ignores hard ones. 

**With Emotional Gate (Our Model):**  
👩‍🍳 Like a human cook:  
- Tastes while cooking.  
- Fixes mistakes (too salty → balance it).  
- Serves with confidence and happiness ❤️.
- Adjusts learning based on “confusion.”
- Pays more attention to tricky digits.
- Learns more human-like: balancing confidence and uncertainty.

✨ Example:  
- **Without Emotion:** Model sees a “3” but mispredicts as “8”. It just accepts the mistake.  
- **With Emotion:** Model says *“Hmm, I’m not confident — let me focus harder on this tricky digit.”* → Improves accuracy.
  
---

## 🎭 Why call it an Emotional Gate if it’s just math?

Of course, this neural network doesn’t truly “feel” like a human. It’s still numbers, weights, and equations.
So why call it an Emotional Gate?

Because in humans, emotions are not just “feelings” — they are signals that guide learning:

Confusion makes us slow down and try harder 😕

Confidence makes us move faster 😎

Curiosity makes us focus more 🔍

We took that principle and encoded it mathematically:

If the network feels uncertain about a digit, the Emotional Gate tells it to adjust learning.

If it feels confident, it passes the decision smoothly.

So the Emotional Gate is like a mathematical metaphor for emotions:

Not real feelings, but a way to make the network behave a little more like us.


---

## 🖼️ Model Flow :  
Input (Pixels)  
      ↓  
 [ Hidden Layer 1 ] → detects edges  
      ↓  
 [ Hidden Layer 2 ] → detects shapes  
      ↓  
 [ Hidden Layer 3 ] → combines patterns ───┐ Thinking Brain 🧠  
      ↓                                    │  
   ┌──────────────────┐                    │  
   │   Emotional      │ → "Does this feel right?" 🍲  
   │       Gate       │ → "Confident or Confused?" 🤔  
   └──────────────────┘                    │  
      ↓  
 [ Output Layer ] → Final Prediction (Digit 0–9)

 ---

 ## Setup :
 
 1) Prerequisites
 
  - Python 3.8+
  - pip

 2) Get the code :

 git clone https://github.com/<your-username>/emotion-aware-mlp-mnist.git 
 cd emotion-aware-mlp-mnist

 3) Create a Virtual Environment :

  python -m venv .venv
  #Windows
  .venv\Scripts\activate
  #macOS/Linux
  source .venv/bin/activate

 4) Install Dependencies :

    pip install -r requirements.txt

 5) Project Structure :

    emotion-aware-mlp-mnist/
├─ data/                 # MNIST .gz (auto-downloaded)
├─ assets/               # checkpoints/plots (optional)
├─ emotion_mlp/
│  ├─ __init__.py
│  ├─ data.py            # download & load MNIST
│  ├─ utils.py           # activations, loss, metrics
│  ├─ model.py           # EmotionAwareMLP (gated MLP)
│  └─ train.py           # training loop
├─ scripts/
│  └─ run_training.py
├─ requirements.txt
└─ .gitignore

 6) Train the Model :

 #Option A: as a module
 python -m emotion_mlp.train

 #Option B: via the script
 python scripts/run_training.py

7) Configuration
- Adjust defaults inside emotion_mlp/train.py:
- epochs (default: 100)
- lr (learning rate, default: 0.01)
- batch_size (default: 64)
- patience (early stopping, default: 8)
- L2 coefficient in model constructor (default: 0.001)

---

## Results

- Early stopping at epoch 72.
- Best metrics:
  - Train Acc: 99.83% (epoch 71)
  - Test/Val Acc: 98.06% (epoch 71)
  - Loss: 0.0137 (epoch 71)

### Training curve (highlights)
- 90%+ accuracy within 3 epochs.
- 96–97% plateau by ~epoch 15.
- Peak around 98.1% test accuracy near epoch 71.
- Early stopping prevented overfitting after later fluctuations.

> The “emotional gate” (sigmoid modulation of the last hidden layer) provides input‑dependent feature weighting, which helped push accuracy beyond the baseline plateau while maintaining generalization.

This project is not about achieving the highest accuracy or building a new model from scratch.
Instead, the main objective is:

To show how an emotion-inspired mechanism (Emotional Gate) can be integrated into a machine learning model.

To demonstrate that even in simple tasks (like MNIST digit classification), an emotion-like layer can improve decision-making.

To provide evidence that human-inspired ideas (like confusion, hesitation, and confidence) can make artificial intelligence more adaptive.


  

    


  


