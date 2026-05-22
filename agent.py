from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS

# 1. Connect to your local Ollama model (must support tool calling)
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)

# 2. Create the internet search tool as a CrewAI-native tool
@tool("DuckDuckGo Search")
def search_tool(query: str) -> str:
    """Search the internet with DuckDuckGo and return the top results."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"{r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
        for r in results
    )

# Agent 1: The Route Planner
route_planner = Agent(
    role='Expert Travel Logistician',
    goal='Find the best travel routes and create Google Maps links for the user.',
    backstory='You are a master of geography and logistics. You always provide clear travel times and direct Google Maps links.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=local_llm
)

# Agent 2: The Accommodation Specialist
hotel_booker = Agent(
    role='Hotel Booking Specialist',
    goal='Search the internet for the best hotels matching the user vibe and budget, prioritizing Booking.com listings.',
    backstory='You are a luxury and budget travel agent. You know how to find the best hotel deals and always provide real names and descriptions of hotels.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=local_llm
)

# We will simulate the data coming from your frontend
user_details = {
    "start_place": "New York",
    "end_place": "Miami",
    "vibe": "chill and beachy",
    "budget": "$150 per night",
    "transport": "driving own car"
}

# Task 1: Route Planning
plan_route_task = Task(
    description=f'''
    The user is traveling from {user_details['start_place']} to {user_details['end_place']} by {user_details['transport']}.
    1. Search for the estimated travel time and distance.
    2. Provide a suggested route.
    3. Generate a Google Maps URL formatted exactly like this:
       https://www.google.com/maps/dir/?api=1&origin={user_details['start_place']}&destination={user_details['end_place']}
    ''',
    expected_output='A summary of the travel route, time, distance, and a clickable Google Maps link.',
    agent=route_planner
)

# Task 2: Hotel Searching
find_hotels_task = Task(
    description=f'''
    The user is going to {user_details['end_place']} with a {user_details['budget']} budget. They want a {user_details['vibe']} vibe.
    1. Use the search tool to find 3 real hotels in {user_details['end_place']} that fit this description.
    2. Try to search specifically for Booking.com listings (e.g., search "best chill hotels in Miami site:booking.com").
    3. Provide the hotel name, a short description, and an estimated price.
    ''',
    expected_output='A list of 3 hotels with descriptions, estimated prices, and recommendations on why they fit the vibe.',
    agent=hotel_booker
)

# Assemble the Crew
travel_crew = Crew(
    agents=[route_planner, hotel_booker],
    tasks=[plan_route_task, find_hotels_task],
    process=Process.sequential  # This makes them work one after the other
)

print("Starting the Travel Agent...")
result = travel_crew.kickoff()

print("==========================================")
print("FINAL TRAVEL PLAN:")
print(result)