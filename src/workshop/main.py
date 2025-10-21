import asyncio
from datetime import date
import json
import logging
import os
from pathlib import Path
import time

from azure.ai.projects import AIProjectClient
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    Agent,
    AgentThread,
    AsyncFunctionTool,
    AsyncToolSet,
    CodeInterpreterTool,
    MessageRole,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from terminal_colors import TerminalColors as tc
from utilities import Utilities

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

load_dotenv()

TENTS_DATA_SHEET_FILE = Path("datasheet/contoso-tents-datasheet.pdf")
API_DEPLOYMENT_NAME = os.getenv("AGENT_MODEL_DEPLOYMENT_NAME")
PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]
AZURE_SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
AZURE_RESOURCE_GROUP_NAME = os.environ["AZURE_RESOURCE_GROUP_NAME"]
AZURE_PROJECT_NAME = os.environ["AZURE_PROJECT_NAME"]
BING_CONNECTION_NAME = os.getenv("BING_CONNECTION_NAME")
MAX_COMPLETION_TOKENS = 4096
MAX_PROMPT_TOKENS = 10240
TEMPERATURE = 0.1
TOP_P = 0.1

utilities = Utilities()

# Project client initialization (outside the context manager for global access)
# Try different client initialization approaches
try:
    # Method 1: Full endpoint approach
    if "/api/projects/" in PROJECT_ENDPOINT:
        project_client = AIProjectClient(
            endpoint=PROJECT_ENDPOINT,
            credential=DefaultAzureCredential(),
        )
        print(f"Using full project endpoint: {PROJECT_ENDPOINT}")
    else:
        # Method 2: Base endpoint approach
        project_client = AIProjectClient(
            endpoint=PROJECT_ENDPOINT,
            credential=DefaultAzureCredential(),
            subscription_id=AZURE_SUBSCRIPTION_ID,
            resource_group_name=AZURE_RESOURCE_GROUP_NAME,
            project_name=AZURE_PROJECT_NAME,
        )
        print(f"Using base endpoint with project details: {PROJECT_ENDPOINT}")
except Exception as e:
    print(f"Error creating project client: {e}")
    # Fallback: try with just endpoint
    project_client = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )
    print("Using fallback client configuration")

# Document search function is defined in setup_agent_tools

INSTRUCTIONS_FILE = "instructions/instructions_function_calling.txt"
# INSTRUCTIONS_FILE = "instructions/instructions_code_interpreter.txt"
INSTRUCTIONS_FILE = "instructions/instructions_file_search.txt"


async def setup_agent_tools() -> AsyncToolSet:
    """Set up all agent tools including document search and sales data."""
    agent_toolset = AsyncToolSet()
    
    try:
        # Set up datasheet directory path
        current_dir = Path(__file__).parent.resolve()
        datasheet_dir = current_dir / "datasheet"
        
        # Ensure datasheet directory exists
        if not datasheet_dir.exists():
            print(f"Creating datasheet directory at: {datasheet_dir}")
            datasheet_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Using existing datasheet directory: {datasheet_dir}")
            
        # List existing documents
        existing_docs = list(datasheet_dir.glob("*.pdf"))
        print(f"Found {len(existing_docs)} existing documents in datasheet folder")
        if existing_docs:
            print("Available documents:")
            for doc in existing_docs:
                print(f"  • {doc.name}")
                
        # First check and sync files from Azure Blob Storage
        print("\n=== Azure Blob Storage Synchronization ===")
        print("Checking environment variables...")
        
        storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT", "").strip()
        storage_key = os.getenv("AZURE_AI", "").strip()
        container_name = os.getenv("AZURE_CONTAINER_NAME", "datasheets").strip()
        
        # Detailed environment variable validation
        env_status = []
        if not storage_account_name:
            env_status.append("❌ AZURE_STORAGE_ACCOUNT environment variable not found or empty")
        else:
            env_status.append(f"✓ AZURE_STORAGE_ACCOUNT: {storage_account_name}")
            
        if not storage_key:
            env_status.append("❌ AZURE_AI environment variable not found or empty")
        else:
            env_status.append(f"✓ AZURE_AI: Key present ({len(storage_key)} characters)")
            
        env_status.append(f"✓ AZURE_CONTAINER_NAME: {container_name} (default: datasheets)")
        
        # Print environment status
        print("\nEnvironment Configuration:")
        for status in env_status:
            print(f"  {status}")
            
        if storage_account_name and storage_key:
            print("\nConnection Details:")
            print(f"  • Storage Account: {storage_account_name}")
            print(f"  • Container: {container_name}")
            print(f"  • Local Directory: {datasheet_dir}")
            print(f"  • Account URL: https://{storage_account_name}.blob.core.windows.net")
            
            try:
                print("\nInitiating blob storage synchronization...")
                downloaded_files = utilities.download_from_blob_storage(
                    storage_account_name=storage_account_name,
                    storage_key=storage_key,
                    container_name=container_name,
                    target_dir=datasheet_dir
                )
                
                if downloaded_files:
                    print(f"\n✅ Successfully synchronized {len(downloaded_files)} files:")
                    for file_path in downloaded_files:
                        print(f"  • {file_path.name}")
                else:
                    print("\n✓ No new or updated files to synchronize")
                    
            except Exception as blob_error:
                print(f"\n❌ Error during blob storage synchronization:")
                print(f"   {str(blob_error)}")
                print("   Continuing with existing local files...")
        else:
            print("\n⚠️  Azure Storage credentials not found")
            print("   Will proceed with existing local files only")
            print("   To enable blob storage sync, set AZURE_STORAGE_ACCOUNT and AZURE_AI environment variables")
            
            # Set up document search function
            async def search_documents(search_term: str) -> str:
                """
                Search through documents in the datasheet folder.
                
                Args:
                    search_term: Text to search for in the documents.
                    
                Returns:
                    JSON string with search results and matching files.
                """
                try:
                    # Validate we're only searching in datasheet directory
                    if not datasheet_dir.exists():
                        return json.dumps({
                            "status": "error",
                            "error": "Datasheet directory not found",
                            "search_term": search_term,
                            "directory": str(datasheet_dir)
                        })
                        
                    # Check if we have any PDF files
                    pdf_files = list(datasheet_dir.glob("*.pdf"))
                    if not pdf_files:
                        return json.dumps({
                            "status": "error",
                            "error": "No PDF documents found in datasheet directory",
                            "search_term": search_term,
                            "directory": str(datasheet_dir)
                        })
                        
                    print(f"\nSearching for: {search_term}")
                    print(f"Location: {datasheet_dir}")
                    print(f"Available documents: {len(pdf_files)}")
                    
                    # Perform the search
                    results = utilities.search_local_files(datasheet_dir, search_term)
                    
                    # Prepare detailed response
                    response = {
                        "matches": len(results),
                        "files": [str(p.name) for p in results],
                        "search_term": search_term,
                        "directory": str(datasheet_dir),
                        "total_documents": len(pdf_files),
                        "status": "success"
                    }
                    
                    if not results:
                        response["suggestion"] = "Try different search terms or check document availability"
                        
                    return json.dumps(response)
                    
                except Exception as e:
                    return json.dumps({
                        "status": "error",
                        "error": str(e),
                        "search_term": search_term,
                        "directory": str(datasheet_dir)
                    })        # Add document search function to toolset
        toolset_functions = {search_documents}
        agent_toolset.add(AsyncFunctionTool(toolset_functions))
        
        # Add code interpreter for visualizations
        agent_toolset.add(CodeInterpreterTool())
        
        print("All agent tools configured successfully")
        return agent_toolset
        
    except Exception as e:
        print(f"Error setting up agent tools: {e}")
        return agent_toolset  # Return toolset even if some setup failed
    # This section has been consolidated into the setup_agent_tools function above
    pass


async def initialize() -> tuple[Agent, AgentThread]:
    """Initialize the agent with document search capabilities."""
    agent = None
    thread = None

    try:
        # Get the current script's directory and resolve the instructions file path
        current_dir = Path(__file__).parent.resolve()
        instructions_path = current_dir / INSTRUCTIONS_FILE
        print(f"Looking for instructions file at: {instructions_path}")
        
        with open(instructions_path, "r", encoding="utf-8", errors="ignore") as file:
            instructions = file.read()

        # Replace the current date placeholder
        instructions = instructions.replace("{current_date}", date.today().strftime("%Y-%m-%d"))

        # Set up all agent tools including document search
        toolset = await setup_agent_tools()

        # Create agent and thread without closing the context manager
        print("Creating agent...")
        agent = project_client.agents.create_agent(
            model=API_DEPLOYMENT_NAME,
            name="company-data-agent",
            instructions=instructions,
            toolset=toolset,
            temperature=TEMPERATURE,
            headers={"x-ms-enable-preview": "true"},
        )
        print(f"Created agent, ID: {agent.id}")

        # Create thread
        print("Creating thread...")
        thread = project_client.agents.threads.create()
        print(f"Created thread, ID: {thread.id}")

        return agent, thread

    except Exception as e:
        logger.error("An error occurred initializing the agent: %s", str(e))
        logger.error("Please ensure you've enabled an instructions file.")
        raise


async def cleanup(agent: Agent, thread: AgentThread) -> None:
    """Cleanup the resources."""
    try:
        project_client.agents.delete_agent(agent.id)
        print(f"Deleted agent: {agent.id}")
    except Exception as e:
        print(f"Error deleting agent: {e}")


async def post_message(thread_id: str, content: str, agent: Agent, thread: AgentThread) -> None:
    """Post a message to the Azure AI Agent Service."""
    try:
        # Get current directory for document operations
        current_dir = Path(__file__).parent.resolve()
        datasheet_dir = current_dir / "datasheet"
        
        # Define document search function in this scope
        async def search_documents(search_term: str) -> str:
            """Search through all documents in the datasheet folder."""
            try:
                results = utilities.search_local_files(datasheet_dir, search_term)
                return json.dumps({
                    "matches": len(results),
                    "files": [str(p.name) for p in results],
                    "search_term": search_term,
                    "status": "success"
                })
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "error": str(e),
                    "search_term": search_term
                })
        
        print(f"Creating message in thread {thread_id}...")
        
        # Create message using project_client directly
        message = project_client.agents.messages.create(
            thread_id=thread_id,
            role="user",
            content=content,
        )
        print(f"Message created: {message.id}")

        print(f"Creating run for agent {agent.id}...")
        # Create and poll run
        run = project_client.agents.runs.create(
            thread_id=thread.id,
            agent_id=agent.id,
        )
        print(f"Run created: {run.id}")
        
        # Enhanced polling with action handling
        import time
        max_iterations = 30  # Reduce max time to 1 minute
        iteration = 0
        last_action_count = 0  # Track number of actions in same state
        last_status = None
        
        while run.status in ("queued", "in_progress", "requires_action") and iteration < max_iterations:
            time.sleep(2)
            iteration += 1
            
            # Track repeated states to detect loops
            if run.status == last_status:
                last_action_count += 1
                if last_action_count > 5:  # Break if same state for too long
                    print("Detected potential infinite loop, breaking...")
                    break
            else:
                last_status = run.status
                last_action_count = 0
            
            try:
                run = project_client.agents.runs.get(thread_id=thread.id, run_id=run.id)
                print(f"Run status: {run.status} (iteration {iteration})")
            except Exception as e:
                print(f"Error getting run status: {e}")
                time.sleep(5)  # Wait longer on error
                continue
            
            # Handle required actions (function calls)
            if run.status == "requires_action" and run.required_action:
                print("Run requires action - handling function calls...")
                
                tool_outputs = []
                try:
                    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
                        print(f"Executing function: {tool_call.function.name}")
                        
                        # Execute the function call
                        args = json.loads(tool_call.function.arguments)
                        
                        if tool_call.function.name == "search_documents":
                            search_term = args.get("search_term", "").strip()
                            print(f"🔍 Searching documents for: {search_term}")
                            result = await search_documents(search_term)
                            
                            # Validate result is proper JSON
                            parsed = json.loads(result)
                            if parsed.get("status") == "success":
                                print(f"\n📂 Search Location: {parsed.get('directory')}")
                                print(f"📊 Available Documents: {parsed.get('total_documents', 0)}")
                                print(f"✓ Found matches in {parsed.get('matches', 0)} documents")
                                
                                if parsed.get('files'):
                                    print("\n📄 Matching Documents:")
                                    for file in parsed.get('files'):
                                        print(f"  • {file}")
                                        
                                if parsed.get('suggestion'):
                                    print(f"\n💡 Suggestion: {parsed.get('suggestion')}")
                            else:
                                print(f"\n⚠️  Search error: {parsed.get('error', 'Unknown error')}")
                                print(f"   Location: {parsed.get('directory', 'unknown')}")
                        else:
                            print(f"❌ Unknown function: {tool_call.function.name}")
                            continue
                            
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": result
                        })
                    
                    # Submit the tool outputs
                    if tool_outputs:
                        print("Submitting tool outputs...")
                        run = project_client.agents.runs.submit_tool_outputs(
                            thread_id=thread.id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )
                        print("Tool outputs submitted successfully")
                except Exception as e:
                    print(f"Error handling tool outputs: {e}")
                    break
        
        if iteration >= max_iterations:
            print("Run timed out after maximum iterations")
            return
            
        print(f"Run finished with status: {run.status}")
        
        if run.status == "failed":
            print(f"Run failed: {run.last_error}")
        elif run.status == "completed":
            # Get the last message from the agent
            try:
                response = project_client.agents.messages.get_last_message_by_role(
                    thread_id=thread_id,
                    role=MessageRole.AGENT,
                )
                if response:
                    print("\nAgent response:")
                    print("\n".join(t.text.value for t in response.text_messages))
                else:
                    print("No response message found")
                
                # Handle file downloads from code interpreter
                try:
                    utilities.download_agent_files(project_client, thread_id)
                except Exception as e:
                    print(f"Error handling file downloads: {e}")
                    
            except Exception as e:
                print(f"Error getting response message: {e}")

    except Exception as e:
        print(f"An error occurred posting the message: {str(e)}")
        import traceback
        traceback.print_exc()


async def main() -> None:
    """
    Main function to run the agent.
    Example questions: Search for waterproof tents, find safety specifications, look for maintenance instructions.
    """
    # Use the project client within a context manager for the entire session
    with project_client:
        agent, thread = await initialize()

        while True:
            # Get user input prompt in the terminal using a pretty shade of green
            print("\n")
            prompt = input(f"{tc.GREEN}Enter your query (type exit to finish): {tc.RESET}")
            if prompt.lower() == "exit":
                break
            if not prompt:
                continue
            await post_message(agent=agent, thread_id=thread.id, content=prompt, thread=thread)

        await cleanup(agent, thread)


if __name__ == "__main__":
    print("Starting async program...")
    asyncio.run(main())
    print("Program finished.")
