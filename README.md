# StratusScope
StratusScope is a language model tool that utilizes HuggingFace's Transformers library and loads two language models of the same architecture and parameter size, consolidates the weights and biases within each layer of both models, examines the aggregate difference between layers, and generates a bar graph detailing which layers have the most difference between each model with matplotlib.

![Figure_2](https://user-images.githubusercontent.com/107712289/232157041-173e8a69-f7e2-439c-b527-61c2da70296d.png)

Use Case - This is an invaluable tool to measure layer differences between a base model and a fine-tune of that model to determine which layers inherited the most change from fine-tuning. With that insight, one may utilize a tool such as LM BlockMerge to transfer knowledge between layers of similar models

Validation Check - Using LM BlockMerge, only layer 15 was 100% transferred from one model to another, and StratusScope accurately depicted the difference.

![Figure_validation](https://user-images.githubusercontent.com/107712289/232157709-3ad6f6db-2f8a-48c3-8f0a-e7c7d138aed3.png)

Colorama is used in the console to provide readable insight into what tensors are inside layers, and what is being contrasted for plotting the resulting graph.

![Figure_validation](https://github.com/Digitous/StratusScope/blob/main/console_out.PNG)


Associated Tool:
LM BlockMerge
https://github.com/TehVenomm/LM_Transformers_BlockMerge
