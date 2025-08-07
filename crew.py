import os
from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
	SerperDevTool,
	ScrapeWebsiteTool
)



@CrewBase
class AiPoweredAdaptiveLearningSystemCrew:
    """AiPoweredAdaptiveLearningSystem crew"""

    
    @agent
    def learning_research_specialist(self) -> Agent:
        
        return Agent(
            config=self.agents_config["learning_research_specialist"],
            tools=[SerperDevTool()],
            reasoning=False,
            inject_date=True,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
        )
    
    @agent
    def content_synthesizer(self) -> Agent:
        
        return Agent(
            config=self.agents_config["content_synthesizer"],
            tools=[ScrapeWebsiteTool()],
            reasoning=False,
            inject_date=True,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
        )
    
    @agent
    def assessment_creator(self) -> Agent:
        
        return Agent(
            config=self.agents_config["assessment_creator"],
            tools=[],
            reasoning=False,
            inject_date=True,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
        )
    
    @agent
    def learning_progress_analyst(self) -> Agent:
        
        return Agent(
            config=self.agents_config["learning_progress_analyst"],
            tools=[],
            reasoning=False,
            inject_date=True,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
        )
    

    
    @task
    def research_learning_topic(self) -> Task:
        return Task(
            config=self.tasks_config["research_learning_topic"],
        )
    
    @task
    def create_structured_learning_content(self) -> Task:
        return Task(
            config=self.tasks_config["create_structured_learning_content"],
        )
    
    @task
    def generate_learning_assessments(self) -> Task:
        return Task(
            config=self.tasks_config["generate_learning_assessments"],
        )
    
    @task
    def create_personalized_learning_plan(self) -> Task:
        return Task(
            config=self.tasks_config["create_personalized_learning_plan"],
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the AiPoweredAdaptiveLearningSystem crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
