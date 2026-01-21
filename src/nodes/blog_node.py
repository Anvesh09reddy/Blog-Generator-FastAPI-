from src.states.blogstate import BlogState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src.states.blogstate import Blog

class BlogNode:
    def __init__(self, llm):
        self.llm = llm
    
    def title_creation(self, state: BlogState):
        if "topic" in state and state["topic"]:
            prompt = """
            You are an expert blog content writer. Use Markdown formatting and 
            generate a blog title about the topic: {topic}. The title should be creative, SEO friendly, short, and catchy.
            Output only the single best title without any additional text.
            """
            system_message = prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
        return {"blog": {"title": response.content}}
        
    def content_generation(self, state: BlogState):
        if "topic" in state and state["topic"]:
            prompt = """
            You are an expert blog content writer. Use Markdown formatting and 
            generate detailed blog content about the topic: {topic}. The content should be engaging and informative.
            """
            system_message = prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
        return {"blog": {"title": state['blog']['title'], "content": response.content}}

    def translation(self, state: BlogState):
        # 1. Initialize the parser
        parser = PydanticOutputParser(pydantic_object=Blog)
        
        # 2. Inject format instructions into the prompt
        translation_prompt = """
        You are an expert translator.
        Translate the following blog content into {current_language}.
        
        Guidelines:
        - Maintain the original tone, style, and formatting (Markdown).
        - Adapt cultural references and idioms to be appropriate for speakers of {current_language}.
        - Ensure the output is valid JSON matching the format below.

        ORIGINAL CONTENT:
        {blog_content}
        
        FORMAT INSTRUCTIONS:
        {format_instructions}
        """
        
        blog_content = state["blog"]["content"]
        
        # 3. Create a PromptTemplate to handle variable injection
        prompt = PromptTemplate(
            template=translation_prompt,
            input_variables=["current_language", "blog_content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # 4. Create a standard chain (Prompt -> LLM -> Parser)
        # This bypasses the specific "Tool Calling" API endpoint that was failing
        chain = prompt | self.llm | parser
        
        try:
            translation_object = chain.invoke({
                "current_language": state["current_language"],
                "blog_content": blog_content
            })
            
            # 5. Return the parsed content
            return {
                "blog": {
                    "title": state['blog']['title'],
                    "content": translation_object.content
                }
            }
        except Exception as e:
            # Fallback if parsing fails (optional, but good for stability)
            print(f"Translation parsing failed: {e}")
            return {"blog": state["blog"]} # Return original if translation fails
    
    def route(self, state: BlogState):
        return {"current_language": state['current_language']}
    
    def route_decision(self, state: BlogState):
        if state["current_language"] == "hindi":
            return "hindi"
        elif state["current_language"] == "french":
            return "french"
        else:
            return state['current_language']