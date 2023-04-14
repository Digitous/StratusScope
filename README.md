# LayerScope
LayerScope is a language model tool that loads two language models of the same architecture and parameter size, consolidates the weights and biases within each layer of both models, examines the aggregate difference between layers, and generates a bar graph detailing which layers have the most difference between the models.

##Use Case
This is an invaluable tool to measure layer differences between a base model and a fine-tune of that model to determine which layers inherited the most change from fine-tuning. With that insight, one may utilize a tool such as LM BlockMerge to transfer knowledge between layers of similar models

![Figure_2](https://user-images.githubusercontent.com/107712289/232157041-173e8a69-f7e2-439c-b527-61c2da70296d.png)
