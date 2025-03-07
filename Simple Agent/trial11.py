def generate_questions(state: AgentState) -> AgentState:
    """Generate questions using the Question Generator tool"""
    try:
        state["thoughts"] += "\n\nPreparing to generate questions..."
        # Extract the topic from the query
        query = state["input"].lower()
        topic = query.replace("generate", "").replace("questions", "").replace("about", "").replace("the history of", "").strip()
        
        # Create a prompt for the LLM to get information about the topic
        info_prompt = PromptTemplate(
            input_variables=["topic"],
            template=(
                "Provide a detailed historical overview of {topic}. Include key dates, facts, and developments. "
                "Focus on major milestones and important events. Make it comprehensive but concise."
            )
        )
        
        state["thoughts"] += f"\n\nGetting information about {topic}..."
        topic_info = llm.invoke(info_prompt.format(topic=topic))
        
        state["thoughts"] += "\n\nUsing Question Generator tool to create structured questions..."
        questions = question_tool.run(f"{topic_info.content} ### Generate 5 questions about the history of {topic}")
        state["thoughts"] += "\n\nQuestions generated successfully. Moving to formatting step..."
        state["questions"] = questions
        state["current_step"] = "questions_generated"
        state["tool_used"] = "questions"
        return state
    except Exception as e:
        state["thoughts"] += f"\n\nError occurred while generating questions: {str(e)}"
        state["final_answer"] = f"Error generating questions: {str(e)}"
        state["current_step"] = "error"
        state["tool_used"] = "questions"
        return state 