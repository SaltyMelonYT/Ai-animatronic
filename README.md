# Powered by Ollama

This is a work-in-progress project to give AI the ability to see, and control itself. As of now I myself do not have the hardware to test such things.
This is an inspired project by [VomitedThoughts](https://www.youtube.com/@vomitedthoughts) and his Mimic and other FNaF related projects.
As of now the AI can **somewhat** see and understand its surroundings.

# Needed models (As of now, these are tested with, but other models should work just fine)
* [llava](https://ollama.com/library/llava) (vision)
* [Deepseek-r1:1.5b](https://ollama.com/library/deepseek-r1:1.5b) (Chat agent, different models with more parameters should be just fine)

# Achieved
* Vision (kind of)

# Needed
* Servo control
* STT and TTS
* Communication between vision and communication (llava --> Llama)
* Less AI generated code as this is far to advanced for me
* Personality
* Memory (If the AI server is shutdown, it can look back at its memories for context) As well as the ability to write memories on its own
