# fortune-dial
This is the repository for the paper "Deal or No Deal (or Who Knows): Forecasting Uncertainty in Conversations using Large Language Models" published in ACL Findings 2024.

Training, inference, and evaluation with our best methods from the paper and the most recent models (Meta's Llama 3.1) is as simple as running the following:
```bash
# Install reqs, you may also need to upgrade transformers for Llama 3.1
pip3 install -r requirements.txt 
# Fine-tune two models: the Implicit Forecaster (SFT) and the Direct Forecaster (RL)
./train-final.sh
# Make the folder where the inference results will be saved
mkdir outputs
# Run test set inference for the implicit forecaster
./eval-final.sh
# Then run test set inference for the direct forecaster
./eval-final-df.sh
# Make the folder where all evaluation statistics will be saved
mkdir frames
# Collect statistics from the inference runs for evaluation
./run/evaluate-final.sh 
# Show the results
python3 -m src.export_llama3 
```

Here's a comparison of these models with some previous runs of GPT-4 (no fine-tuning). We use data priors and temperature scaling for both models (see paper for details).

| model                 | alg          | instances          | Brier Score        |
|:----------------------|:-------------|:-------------------|:----------|
|Llama-3.1-8B-Instruct  | DF RL interp | awry               | 0.255467  |
|                       |              | casino             | 0.216955  |
|                       |              | cmv                | 0.261726  |               
|                       |              | deals              | 0.174899  |
|                       |              | deleted            | 0.255129  |
|                       |              | donations          | 0.251880  |
|                       |              | supreme            | 0.231955  |
|Llama-3.1-8B-Instruct  | IF SFT       | awry               | 0.220083  |
|                       |              | casino             | 0.196558  |
|                       |              | cmv                | 0.207542  |               
|                       |              | deals              | 0.118853  |
|                       |              | deleted            | 0.114553  |
|                       |              | donations          | 0.238121  |
|                       |              | supreme            | 0.223060  |
|OpenAI GPT 4           | None         | awry               | 0.247775  |
|                       |              | casino             | 0.204828  |
|                       |              | cmv                | 0.230229  |               
|                       |              | deals              | 0.132760  |
|                       |              | deleted            | 0.169750  |
|                       |              | donations          | 0.262453  |
|                       |              | supreme            | 0.230321  |
 

The fine-tuned models are available on huggingface, if you prefer to directly use those without any customization (links forthcoming). You can use other scripts in the ```run``` directory to reproduce results from the paper (same order of runs as above) and use ```python3 -m src.export``` to show these results. Plenty of other customization is possible as well and don't hesitate to reach out with questions!

Up to date contact details are available at my website: https://anthonysicilia.tech