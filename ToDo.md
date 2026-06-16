# ToDo 

### Supervised Fine-Tuning (PEFT): in progress

- fine-tune Gemma-4-E4B-it model 
- zero-shot
- implicit prompt template
- quantisation?
- 4-fold CV?

### API Inference: in progress

- fill LM_API class (or adapt LM class) for API calls
- settings to run in configs_todo
- 4-fold CV
- run both unedited settings

Report: 
- unedited did not work:
  - had to switch a demonstration mode, 
  - embedding mode,
  - retrieval mode and 
  - allow for at least 16 tokens (OpenAI minimum)
- ran config 1 + config 2 as documented in the config.yaml