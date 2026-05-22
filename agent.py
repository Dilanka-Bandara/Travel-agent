from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS
from pydantic import BaseModel, Field
from typing import List

# ==========================================
# 1. DEFINE NESTED STRUCTURED OUTPUT
# ==========================================
class DailyPlan(BaseModel):
    day_number: int = Field(description="Which day of the trip this is")
    route_segment: str = Field(description="The starting and ending town for this specific day (e.g., 'Jaffna to Anuradhapura')")
    activities: List[str] = Field(description="2 to 3 specific activities or attractions along this day's segment that match the user vibe")
    hotel_name: str = Field(description="Name of the real recommended hotel for this night's stop")
    hotel_price: str = Field(description="Estimated price per night")

class RouteOption(BaseModel):
    path_name: str = Field(description="The theme or name of this route (e.g., 'The Historical Inland Route')")
    full_google_maps_url: str = Field(description="Google Maps URL with all overnight waypoints included")
    daily_breakdown: List[DailyPlan] = Field(description="A day-by-day breakdown of the journey")
    why_it_fits: str = Field(description="Why this overall route fits the user's custom description")

class CustomTripItinerary(BaseModel):
    trip_summary: str = Field(description="A brief overview of how the agent customized these choices")
    routes: List[RouteOption] = Field(description="A list containing exactly 3 distinct route options")


# ==========================================
# 2. SETUP LLM AND TOOLS
# ==========================================
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
    timeout=600 # Increased timeout because multi-day planning takes time
)

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


# ==========================================
# 3. DEFINE THE AGENTS
# ==========================================
route_planner = Agent(
    role='Expert Travel Route Architect',
    goal='Divide a journey into daily segments based on user duration, creating 3 distinct paths.',
    backstory='You are a master road-trip planner. You know how to pace a trip. If a user has 3 days, you find the perfect overnight stopping towns at the 1/3 and 2/3 marks of the journey.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=local_llm
)

experience_specialist = Agent(
    role='Local Experience & Accommodation Specialist',
    goal='Find activities and hotels along specific daily segments created by the Route Architect.',
    backstory='You are a local guide. You know exactly what activities to do along a stretch of road, and the best hotels to sleep at when the day is done. You tailor everything strictly to the user vibe.',
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=local_llm
)


# ==========================================
# 4. GET USER INPUTS
# ==========================================
print("\n🌍 Welcome to the AI Multi-Day Trip Designer!")
print("-" * 70)

user_details = {
    "start_place": input("📍 Starting Point? (e.g., Jaffna): "),
    "end_place": input("🏁 Final Destination? (e.g., Colombo): "),
    "duration": input("⏳ How many days for the journey? (e.g., 3): "),
    "vibe": input("✨ Main vibe? (e.g., extreme adventure, historical, relaxed): "),
    "budget": input("💰 Hotel budget per night? (e.g., $50, luxury): "),
    "transport": input("🚗 Mode of transport? (e.g., driving own car): "),
    "trip_description": input("📝 Describe your dream trip: ")
}

print("\n🤖 Calculating pacing, researching activities, and finding hotels... (This may take a few minutes)\n")
print("-" * 70)


# ==========================================
# 5. DEFINE THE TASKS
# ==========================================
plan_route_task = Task(
    description=f'''
    The user is driving from {user_details['start_place']} to {user_details['end_place']} over {user_details['duration']} days by {user_details['transport']}.
    Their vibe: "{user_details['vibe']}".

    Design 3 DIFFERENT travel paths (e.g., Coastal, Central, Fast).
    
    For EACH of the 3 paths:
    1. Calculate where they should stop to sleep each night to break up the journey evenly over {user_details['duration']} days.
    2. Provide the towns for those overnight stops.
    3. Generate ONE master Google Maps URL for the whole path linking the start, waypoints, and end.
    ''',
    expected_output='A strategic pacing guide for 3 different paths, detailing exactly which towns to sleep in each night.',
    agent=route_planner
)

build_itinerary_task = Task(
    description=f'''
    Read the pacing guide from the Route Architect.
    The user wants a "{user_details['vibe']}" experience and specifically noted: "{user_details['trip_description']}".
    Budget: {user_details['budget']} per night.
    
    For EACH of the 3 paths, you must create a daily breakdown for all {user_details['duration']} days:
    1. For each day's driving segment, use your search tool to find 2-3 real activities along the way that perfectly match the vibe.
    2. For each night's stopping town, search for 1 real hotel matching the budget.
    
    Format everything into the CustomTripItinerary schema perfectly.
    ''',
    expected_output=f'A fully populated CustomTripItinerary containing 3 paths, with {user_details['duration']} days of activities and hotels for each path.',
    agent=experience_specialist,
    output_pydantic=CustomTripItinerary
)


# ==========================================
# 6. ASSEMBLE AND EXECUTE
# ==========================================
travel_crew = Crew(
    agents=[route_planner, experience_specialist],
    tasks=[plan_route_task, build_itinerary_task],
    process=Process.sequential
)

crew_output = travel_crew.kickoff()


# ==========================================
# 7. PRINT NESTED OUTPUT BEAUTIFULLY
# ==========================================
try:
    final_plan: CustomTripItinerary = crew_output.pydantic
    
    print("\n" + "="*80)
    print(f"🗺️  YOUR {user_details['duration']}-DAY CUSTOM TRAVEL PLAN")
    print("="*80)
    print(f"\n🤖 System Overview: {final_plan.trip_summary}\n")

    for i, route in enumerate(final_plan.routes, 1):
        print("=" * 80)
        print(f"🌟 OPTION {i}: {route.path_name.upper()}")
        print(f"🗺️  Full Route Map : {route.full_google_maps_url}")
        print(f"✨ Why it fits   : {route.why_it_fits}")
        print("-" * 80)
        
        for day in route.daily_breakdown:
            print(f"   📅 DAY {day.day_number} | Drive: {day.route_segment}")
            print("      🎯 Activities:")
            for act in day.activities:
                print(f"         - {act}")
            print(f"      🏨 Night Stay: {day.hotel_name} ({day.hotel_price})")
            print("")
            
except Exception as e:
    print("\n⚠️ The model struggled to compile all the data into the strict JSON format.")
    print("Here is the raw text output it managed to generate:")
    print(crew_output.raw)