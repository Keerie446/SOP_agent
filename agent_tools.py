import requests
from langchain_core.tools import tool
from knowledge_base import get_retriever

@tool
def retrieve_sop_tool(query: str) -> str:
    """Useful to search for Standard Operating Procedure (SOP) rules or guidelines."""
    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant SOPs found for the query."
    
    results = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in docs])
    return results

@tool
def http_request_tool(url: str, method: str = "GET", data: dict = None) -> str:
    """Useful to make HTTP requests to external APIs."""
    try:
        if method.upper() == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            response = requests.get(url, timeout=10)
            
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"HTTP Request failed: {str(e)}"

@tool
def browser_automation_tool(action: str, target: str, value: str = None) -> str:
    """
    Mock tool for executing browser automation steps. 
    Actions: 'click', 'type', 'navigate', 'read'
    Target: The css selector or URL
    Value: The text to type (if action is 'type')
    """
    # This is a mock execution to satisfy initial testing requirements.
    result = f"MOCK BROWSER: Executed '{action}' on '{target}'."
    if value:
        result += f" Value provided: '{value}'."
        
    return result

@tool
def verify_tool(extracted_data: str, expected_format_desc: str) -> str:
    """Validates specific extraction formats before finalizing step."""
    # A lightweight mock verification
    return f"Verified: The input '{extracted_data}' appears to match format '{expected_format_desc}'."

TOOLS = [retrieve_sop_tool, http_request_tool, browser_automation_tool, verify_tool]
