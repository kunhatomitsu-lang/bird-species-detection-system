# Wetlands Birds Detector

AI web application to identify 6 distinct wetland bird species in real time using a fine-tuned EfficientNet-B0 model and a Gradio user interface.

---

## Supported Bird Species
The model is trained to recognize the following 6 species with a strict 70% confidence verification gate:
1. Black-crowned Night Heron
2. Common Kingfisher
3. Greater Cormorant
4. Hooded Merganser
5. Mallard
6. Marsh Sandpiper

---

## How to Run and Deploy the UI

### 1. Directly on Hugging Face Spaces (Online)
To host and run this application live on Hugging Face Spaces, follow these steps:
1. **Create a Space:** Go to Hugging Face, click on your profile, select **New Space**, choose a name, and select **Gradio** as the Space SDK.
2. **Clone the Space Repository:** Clone your newly created Hugging Face Space repository to your local computer using the terminal:
```bash
git clone [https://github.com/kunhatomitsu-lang/bird-species-detection-system.git](https://github.com/kunhatomitsu-lang/bird-species-detection-system.git)
   cd bird-species-detection-system.git
```
3. **Add Production Files:** Place the required production files (app.py, fold_0_best.pth, requirements.txt, .gitattributes, and this README.md) directly into the root folder of the Space.
4. **Push to Hugging Face:** Stage, commit, and push your files to deploy the application:
```bash
git add -A
git commit -m "Deploy Wetlands Birds Detector UI"
git push origin main
```
Hugging Face will automatically build the environment and launch the live web application.

### 2. Locally on Your Machine
If you want to run the application locally from your cloned repository branch, follow these steps:

1. Clone the repository and navigate to the branch:
   ```bash
   git clone [https://github.com/kunhatomitsu-lang/bird-species-detection-system.git](https://github.com/kunhatomitsu-lang/bird-species-detection-system.git)
   cd bird-species-detection-system.git
   git checkout bird-detection-interface
   ```
2. **Install Dependencies:** Run the following command in your terminal to install all required packages:
```bash
pip install -r requirements.txt
```
3. **Launch the Application:** Start the local server by executing:
```bash
python app.py
```
Open the local URL provided in your terminal (typically https://127.0.0.1:7860) to interact with the UI.

## Safety and Confidence Gate

To prevent hallucination or false matches on unrelated photos, the application enforces a 60% confidence threshold:
1. **>=60% Confidence:** Displays the matches species name and full probability breakdown.
2. **670% Confidence:** Triggers a safety fallback and flags the input as "Unrecognized Species".
