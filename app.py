import chainlit as cl
from langchain_core.messages import HumanMessage
from agent_graph import app as graph_app

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [])
    cl.user_session.set("plan", [])
    cl.user_session.set("current_step", 0)
    
    await cl.Message(
        content="Welcome to the SOP Agent! Please upload SOP documents into the `./sops` directory and run `python knowledge_base.py` before querying me. How can I assist you with your procedures today?"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve current state from user session
    messages = cl.user_session.get("messages")
    plan = cl.user_session.get("plan")
    current_step = cl.user_session.get("current_step")
    
    # Append the new human message
    messages.append(HumanMessage(content=message.content))
    
    # Construct the state dictionary for LangGraph
    state = {
        "messages": messages,
        "current_sop": "",
        "plan": plan,
        "current_step": current_step,
        "status": "processing"
    }

    # Set up UI elements for streaming visibility
    msg = cl.Message(content="")
    await msg.send()
    
    output_content = ""
    
    try:
        # Stream events from LangGraph
        async for event in graph_app.astream_events(state, version="v2"):
            kind = event["event"]
            name = event["name"]
            
            # Stream text chunks from the LLM (Executor node)
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    output_content += chunk.content
                    await msg.stream_token(chunk.content)
                    
            # Show when a tool starts
            elif kind == "on_tool_start":
                tool_name = name
                tool_input = event["data"].get("input", {})
                await cl.Message(content=f"🛠️ *Executing Tool:* `{tool_name}` with input: `{tool_input}`").send()
                
            # Show when a tool finishes
            elif kind == "on_tool_end":
                tool_name = name
                # Truncate output if it's too long
                tool_output_str = str(event["data"].get("output", ""))
                tool_output = tool_output_str[:500] + ("..." if len(tool_output_str) > 500 else "")
                await cl.Message(content=f"✅ *Tool `{tool_name}` Finished.* Output snippet:\\n```\\n{tool_output}\\n```").send()
                
        # Update the UI message and the user session state
        await msg.update()
        
        # For simplistic multi-turn, we retrieve the final state by running it once more or just relying on what we collected.
        # In a full LangGraph setup with memory, we'd persist the thread state. 
        # Here we simulate keeping the conversation growing.
        final_state = await graph_app.ainvoke(state)
        
        # Update session memory
        cl.user_session.set("messages", final_state["messages"])
        cl.user_session.set("plan", final_state.get("plan", []))
        cl.user_session.set("current_step", final_state.get("current_step", 0))
    except Exception as e:
        await cl.Message(content=f"⚠️ *An error occurred during execution:* {str(e)}").send()
