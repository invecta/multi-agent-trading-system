"""
Advanced Async Workflow Orchestration System
Implements sophisticated parallel agent execution with dependency management and error handling
"""
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import uuid

from multi_agent_framework import BaseAgent, AgentOutput, AgentStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class ExecutionPriority(Enum):
    """Execution priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class WorkflowNode:
    """Workflow node representing an agent"""
    agent_id: str
    agent: BaseAgent
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    priority: ExecutionPriority = ExecutionPriority.MEDIUM
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    status: WorkflowStatus = WorkflowStatus.PENDING
    output: Optional[AgentOutput] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class WorkflowExecution:
    """Workflow execution context"""
    execution_id: str
    workflow_name: str
    nodes: Dict[str, WorkflowNode]
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime]
    total_execution_time: float
    completed_nodes: Set[str]
    failed_nodes: Set[str]
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    error_log: List[str]
    performance_metrics: Dict[str, Any]

class AsyncWorkflowOrchestrator:
    """Advanced async workflow orchestrator with parallel execution"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_concurrent_agents = self.config.get('max_concurrent_agents', 5)
        self.default_timeout = self.config.get('default_timeout', 30.0)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.enable_circuit_breaker = self.config.get('enable_circuit_breaker', True)
        
        # Execution tracking
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        self.agent_registry: Dict[str, BaseAgent] = {}
        
        # Performance monitoring
        self.performance_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'agent_performance': {}
        }
        
        # Circuit breaker for failed agents
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Thread pool for CPU-intensive tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_concurrent_agents)
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agent_registry[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
    
    def create_workflow(self, workflow_name: str, agent_configs: List[Dict[str, Any]]) -> str:
        """Create a new workflow"""
        execution_id = str(uuid.uuid4())
        
        # Create workflow nodes
        nodes = {}
        for config in agent_configs:
            agent_id = config['agent_id']
            if agent_id not in self.agent_registry:
                raise ValueError(f"Agent {agent_id} not registered")
            
            node = WorkflowNode(
                agent_id=agent_id,
                agent=self.agent_registry[agent_id],
                dependencies=set(config.get('dependencies', [])),
                priority=ExecutionPriority(config.get('priority', ExecutionPriority.MEDIUM.value)),
                timeout=config.get('timeout', self.default_timeout),
                max_retries=config.get('max_retries', 3)
            )
            nodes[agent_id] = node
        
        # Build dependency graph
        self._build_dependency_graph(nodes)
        
        # Create workflow execution
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_name=workflow_name,
            nodes=nodes,
            status=WorkflowStatus.PENDING,
            start_time=datetime.now(),
            end_time=None,
            total_execution_time=0.0,
            completed_nodes=set(),
            failed_nodes=set(),
            input_data={},
            output_data={},
            error_log=[],
            performance_metrics={}
        )
        
        self.active_executions[execution_id] = execution
        logger.info(f"Created workflow {workflow_name} with execution ID: {execution_id}")
        
        return execution_id
    
    def _build_dependency_graph(self, nodes: Dict[str, WorkflowNode]):
        """Build dependency graph between nodes"""
        for node_id, node in nodes.items():
            for dep_id in node.dependencies:
                if dep_id in nodes:
                    node.dependencies.add(dep_id)
                    nodes[dep_id].dependents.add(node_id)
                else:
                    logger.warning(f"Dependency {dep_id} not found for node {node_id}")
    
    async def execute_workflow(self, execution_id: str, input_data: Dict[str, Any]) -> WorkflowExecution:
        """Execute a workflow"""
        if execution_id not in self.active_executions:
            raise ValueError(f"Workflow execution {execution_id} not found")
        
        execution = self.active_executions[execution_id]
        execution.status = WorkflowStatus.RUNNING
        execution.input_data = input_data
        
        logger.info(f"Starting workflow execution: {execution.workflow_name}")
        
        try:
            # Execute workflow
            await self._execute_workflow_nodes(execution)
            
            # Calculate final metrics
            execution.end_time = datetime.now()
            execution.total_execution_time = (execution.end_time - execution.start_time).total_seconds()
            
            # Determine final status
            if execution.failed_nodes:
                execution.status = WorkflowStatus.FAILED
                self.performance_stats['failed_executions'] += 1
            else:
                execution.status = WorkflowStatus.COMPLETED
                self.performance_stats['successful_executions'] += 1
            
            # Update performance stats
            self._update_performance_stats(execution)
            
            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution_id]
            
            logger.info(f"Workflow execution completed: {execution.workflow_name} - Status: {execution.status.value}")
            
            return execution
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_log.append(f"Workflow execution failed: {str(e)}")
            execution.end_time = datetime.now()
            execution.total_execution_time = (execution.end_time - execution.start_time).total_seconds()
            
            self.performance_stats['failed_executions'] += 1
            self.execution_history.append(execution)
            del self.active_executions[execution_id]
            
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def _execute_workflow_nodes(self, execution: WorkflowExecution):
        """Execute workflow nodes with dependency management"""
        nodes = execution.nodes
        completed_nodes = execution.completed_nodes
        failed_nodes = execution.failed_nodes
        
        # Create execution queue with priority
        execution_queue = asyncio.PriorityQueue()
        
        # Add initial nodes (no dependencies) to queue
        for node_id, node in nodes.items():
            if not node.dependencies:
                priority = node.priority.value
                await execution_queue.put((priority, node_id, node))
        
        # Execute nodes
        active_tasks = {}
        max_concurrent = min(self.max_concurrent_agents, len(nodes))
        
        while not execution_queue.empty() or active_tasks:
            # Start new tasks if we have capacity
            while len(active_tasks) < max_concurrent and not execution_queue.empty():
                try:
                    priority, node_id, node = await execution_queue.get_nowait()
                    
                    # Check if node can execute (dependencies completed)
                    if node.dependencies.issubset(completed_nodes):
                        task = asyncio.create_task(self._execute_node(node, execution))
                        active_tasks[node_id] = task
                    else:
                        # Put back in queue
                        await execution_queue.put((priority, node_id, node))
                        break
                        
                except asyncio.QueueEmpty:
                    break
            
            # Wait for at least one task to complete
            if active_tasks:
                done, pending = await asyncio.wait(
                    active_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task in done:
                    node_id = None
                    for nid, t in active_tasks.items():
                        if t == task:
                            node_id = nid
                            break
                    
                    if node_id:
                        del active_tasks[node_id]
                        
                        try:
                            result = await task
                            if result:
                                completed_nodes.add(node_id)
                                
                                # Add dependent nodes to queue
                                node = nodes[node_id]
                                for dependent_id in node.dependents:
                                    if dependent_id not in completed_nodes and dependent_id not in failed_nodes:
                                        dependent_node = nodes[dependent_id]
                                        priority = dependent_node.priority.value
                                        await execution_queue.put((priority, dependent_id, dependent_node))
                            else:
                                failed_nodes.add(node_id)
                                
                        except Exception as e:
                            logger.error(f"Error processing completed task for {node_id}: {e}")
                            failed_nodes.add(node_id)
        
        # Check if all nodes completed successfully
        if failed_nodes:
            execution.error_log.append(f"Failed nodes: {list(failed_nodes)}")
    
    async def _execute_node(self, node: WorkflowNode, execution: WorkflowExecution) -> bool:
        """Execute a single workflow node"""
        try:
            # Check circuit breaker
            if self.enable_circuit_breaker and self._is_circuit_open(node.agent_id):
                logger.warning(f"Circuit breaker open for agent {node.agent_id}")
                node.status = WorkflowStatus.FAILED
                node.error = "Circuit breaker open"
                return False
            
            # Update node status
            node.status = WorkflowStatus.RUNNING
            node.started_at = datetime.now()
            
            # Prepare input data for the node
            input_data = self._prepare_node_input(node, execution)
            
            # Execute agent with timeout
            start_time = time.time()
            
            try:
                output = await asyncio.wait_for(
                    node.agent.process(input_data),
                    timeout=node.timeout
                )
                
                execution_time = time.time() - start_time
                node.execution_time = execution_time
                node.output = output
                node.status = WorkflowStatus.COMPLETED
                node.completed_at = datetime.now()
                
                # Update execution output data
                execution.output_data[node.agent_id] = output.data
                
                # Reset circuit breaker on success
                if self.enable_circuit_breaker:
                    self._reset_circuit_breaker(node.agent_id)
                
                logger.info(f"Node {node.agent_id} completed successfully in {execution_time:.2f}s")
                return True
                
            except asyncio.TimeoutError:
                node.status = WorkflowStatus.FAILED
                node.error = f"Timeout after {node.timeout}s"
                execution.error_log.append(f"Node {node.agent_id} timed out")
                
                # Increment circuit breaker
                if self.enable_circuit_breaker:
                    self._increment_circuit_breaker(node.agent_id)
                
                return False
                
        except Exception as e:
            node.status = WorkflowStatus.FAILED
            node.error = str(e)
            execution.error_log.append(f"Node {node.agent_id} failed: {str(e)}")
            
            # Increment circuit breaker
            if self.enable_circuit_breaker:
                self._increment_circuit_breaker(node.agent_id)
            
            logger.error(f"Node {node.agent_id} execution failed: {e}")
            return False
    
    def _prepare_node_input(self, node: WorkflowNode, execution: WorkflowExecution) -> Dict[str, Any]:
        """Prepare input data for a node based on dependencies"""
        input_data = execution.input_data.copy()
        
        # Add outputs from dependency nodes
        for dep_id in node.dependencies:
            if dep_id in execution.output_data:
                dep_output = execution.output_data[dep_id]
                input_data[dep_id] = dep_output
        
        return input_data
    
    def _is_circuit_open(self, agent_id: str) -> bool:
        """Check if circuit breaker is open for an agent"""
        if agent_id not in self.circuit_breakers:
            return False
        
        circuit = self.circuit_breakers[agent_id]
        failure_count = circuit.get('failure_count', 0)
        last_failure = circuit.get('last_failure_time')
        
        if failure_count >= 5:  # Circuit opens after 5 failures
            if last_failure:
                # Circuit stays open for 5 minutes
                if (datetime.now() - last_failure).total_seconds() < 300:
                    return True
                else:
                    # Reset circuit breaker
                    self._reset_circuit_breaker(agent_id)
        
        return False
    
    def _increment_circuit_breaker(self, agent_id: str):
        """Increment circuit breaker failure count"""
        if agent_id not in self.circuit_breakers:
            self.circuit_breakers[agent_id] = {'failure_count': 0}
        
        self.circuit_breakers[agent_id]['failure_count'] += 1
        self.circuit_breakers[agent_id]['last_failure_time'] = datetime.now()
    
    def _reset_circuit_breaker(self, agent_id: str):
        """Reset circuit breaker for an agent"""
        if agent_id in self.circuit_breakers:
            self.circuit_breakers[agent_id]['failure_count'] = 0
            self.circuit_breakers[agent_id]['last_failure_time'] = None
    
    def _update_performance_stats(self, execution: WorkflowExecution):
        """Update performance statistics"""
        self.performance_stats['total_executions'] += 1
        
        # Update average execution time
        total_time = self.performance_stats['average_execution_time'] * (self.performance_stats['total_executions'] - 1)
        total_time += execution.total_execution_time
        self.performance_stats['average_execution_time'] = total_time / self.performance_stats['total_executions']
        
        # Update agent performance
        for node_id, node in execution.nodes.items():
            if node_id not in self.performance_stats['agent_performance']:
                self.performance_stats['agent_performance'][node_id] = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'average_execution_time': 0.0,
                    'failure_rate': 0.0
                }
            
            agent_perf = self.performance_stats['agent_performance'][node_id]
            agent_perf['total_executions'] += 1
            
            if node.status == WorkflowStatus.COMPLETED:
                agent_perf['successful_executions'] += 1
            
            # Update average execution time
            total_time = agent_perf['average_execution_time'] * (agent_perf['total_executions'] - 1)
            total_time += node.execution_time
            agent_perf['average_execution_time'] = total_time / agent_perf['total_executions']
            
            # Update failure rate
            agent_perf['failure_rate'] = 1.0 - (agent_perf['successful_executions'] / agent_perf['total_executions'])
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
        else:
            # Look in history
            execution = next((e for e in self.execution_history if e.execution_id == execution_id), None)
        
        if not execution:
            return None
        
        return {
            'execution_id': execution.execution_id,
            'workflow_name': execution.workflow_name,
            'status': execution.status.value,
            'start_time': execution.start_time.isoformat(),
            'end_time': execution.end_time.isoformat() if execution.end_time else None,
            'total_execution_time': execution.total_execution_time,
            'completed_nodes': list(execution.completed_nodes),
            'failed_nodes': list(execution.failed_nodes),
            'error_log': execution.error_log,
            'node_statuses': {
                node_id: {
                    'status': node.status.value,
                    'execution_time': node.execution_time,
                    'error': node.error
                }
                for node_id, node in execution.nodes.items()
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.performance_stats.copy()
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        execution.status = WorkflowStatus.CANCELLED
        execution.end_time = datetime.now()
        execution.total_execution_time = (execution.end_time - execution.start_time).total_seconds()
        
        # Move to history
        self.execution_history.append(execution)
        del self.active_executions[execution_id]
        
        logger.info(f"Cancelled execution: {execution_id}")
        return True
    
    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get list of active executions"""
        return [
            {
                'execution_id': execution_id,
                'workflow_name': execution.workflow_name,
                'status': execution.status.value,
                'start_time': execution.start_time.isoformat(),
                'completed_nodes': len(execution.completed_nodes),
                'total_nodes': len(execution.nodes)
            }
            for execution_id, execution in self.active_executions.items()
        ]
    
    def cleanup_old_executions(self, max_age_hours: int = 24):
        """Clean up old execution history"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Remove old executions from history
        self.execution_history = [
            execution for execution in self.execution_history
            if execution.start_time > cutoff_time
        ]
        
        logger.info(f"Cleaned up executions older than {max_age_hours} hours")

class WorkflowBuilder:
    """Workflow builder for creating complex workflows"""
    
    def __init__(self, orchestrator: AsyncWorkflowOrchestrator):
        self.orchestrator = orchestrator
        self.workflow_config = []
    
    def add_agent(self, agent_id: str, dependencies: List[str] = None, 
                 priority: ExecutionPriority = ExecutionPriority.MEDIUM,
                 timeout: float = 30.0, max_retries: int = 3) -> 'WorkflowBuilder':
        """Add an agent to the workflow"""
        self.workflow_config.append({
            'agent_id': agent_id,
            'dependencies': dependencies or [],
            'priority': priority.value,
            'timeout': timeout,
            'max_retries': max_retries
        })
        return self
    
    def build(self, workflow_name: str) -> str:
        """Build and create the workflow"""
        return self.orchestrator.create_workflow(workflow_name, self.workflow_config)

# Example usage and testing
async def main():
    """Example usage of the async workflow orchestrator"""
    
    # Create orchestrator
    orchestrator = AsyncWorkflowOrchestrator({
        'max_concurrent_agents': 3,
        'default_timeout': 30.0,
        'enable_circuit_breaker': True
    })
    
    # Register agents (simplified for demo)
    from multi_agent_framework import MarketDataAgent, TechnicalAnalysisAgent, RiskManagerAgent
    
    orchestrator.register_agent(MarketDataAgent())
    orchestrator.register_agent(TechnicalAnalysisAgent())
    orchestrator.register_agent(RiskManagerAgent())
    
    # Build workflow
    builder = WorkflowBuilder(orchestrator)
    execution_id = builder.add_agent('market_data_agent') \
                          .add_agent('technical_analysis_agent', ['market_data_agent']) \
                          .add_agent('risk_manager_agent', ['technical_analysis_agent']) \
                          .build('trading_workflow')
    
    # Execute workflow
    input_data = {
        'symbols': ['AAPL', 'GOOGL'],
        'timeframes': ['1d', '1h']
    }
    
    try:
        result = await orchestrator.execute_workflow(execution_id, input_data)
        print(f"Workflow completed: {result.status.value}")
        print(f"Execution time: {result.total_execution_time:.2f}s")
        print(f"Completed nodes: {len(result.completed_nodes)}")
        print(f"Failed nodes: {len(result.failed_nodes)}")
        
    except Exception as e:
        print(f"Workflow failed: {e}")
    
    # Print performance stats
    stats = orchestrator.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"Total executions: {stats['total_executions']}")
    print(f"Success rate: {stats['successful_executions'] / max(stats['total_executions'], 1):.2%}")
    print(f"Average execution time: {stats['average_execution_time']:.2f}s")

if __name__ == "__main__":
    asyncio.run(main())
