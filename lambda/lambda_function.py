from ask_sdk_core.dispatch_components import AbstractExceptionHandler, AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
import ask_sdk_core.utils as ask_utils
import boto3
import logging
import json

from utils import load_system_prompt  # Import function from utils file
"""
The system_prompt_file_path = "system_prompt.txt" is used to define the system prompt (also called system message) for your AI assistant.

Purpose:

System prompt tells the AI model how to behave, what role to play, and what guidelines to follow

It's the initial instruction that shapes the AI's personality and response style

For an Alexa skill, it might define how the AI should respond to voice queries


"""
# Load system prompt from file
system_prompt_file_path = "system_prompt.txt"
system_prompt = load_system_prompt(system_prompt_file_path)

# Initialize Bedrock client
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # or your preferred model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        speak_output = "Alexa gen AI mode activated"
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["chat_history"] = []
        return handler_input.response_builder.speak(speak_output).ask(speak_output).response


class GenAiQueryIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("GenAiQueryIntent")(handler_input)

    def handle(self, handler_input):
        query = handler_input.request_envelope.request.intent.slots["query"].value
        session_attr = handler_input.attributes_manager.session_attributes
        if "chat_history" not in session_attr:
            session_attr["chat_history"] = []
        response = generate_GenAi_response(session_attr["chat_history"], query)
        session_attr["chat_history"].append((query, response))
        return handler_input.response_builder.speak(response).ask("Any other questions?").response


class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        speak_output = "Leaving Alexa Gen AI mode"
        return handler_input.response_builder.speak(speak_output).response


class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        speak_output = "Sorry, I had trouble doing what you asked. Please try again."
        return handler_input.response_builder.speak(speak_output).ask(speak_output).response


def generate_GenAi_response(chat_history, new_question):
    messages = [{"role": "user", "content": f"{system_prompt}\n\n{new_question}"}]
    for question, answer in chat_history[-5:]:
        messages[0]["content"] += f"\n\nPrevious Q: {question}\nPrevious A: {answer}"
    
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 300,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.95
    }
    
    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except Exception as e:
        return f"Error generating response: {str(e)}"


sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(GenAiQueryIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
