Gemini
LLMs for Evolutionary Max-SAT
https://gemini.google.com/share/662abe5190a3
נוצר על ידי Pro 25 בדצמבר 2025 בשעה 10:09
מועד הפרסום: 25 בדצמבר 2025 בשעה 10:14
I wish to use LLMs capabilities as part of evolutionary technique for solving Max-SAT is there anything in GitHub for this?

While there is currently no single, famous "plug-and-play" GitHub repository dedicated exclusively to LLM-based Evolutionary Max-SAT, the research field is very active.

You can achieve this by adapting general-purpose LLM-driven evolutionary frameworks or by building a custom loop where the LLM acts as the mutation/crossover operator.

1. Best Existing GitHub Repositories (Adaptable)
These repositories are designed to evolve heuristics or solutions for combinatorial problems (like Bin Packing or TSP) and can be easily retargeted for Max-SAT.

Repository	Description	Suitability for Max-SAT
google-deepmind/funsearch	The official code for the Nature paper "Mathematical discoveries from program search with LLMs."	High. You can replace their "Bin Packing" evaluator with a Max-SAT evaluator (e.g., PySAT).
FeiLiu36/EoH	"Evolution of Heuristics" - uses LLMs to evolve both the thought (idea) and the code of a heuristic.	High. Excellent for evolving a Max-SAT solver rather than just a single solution instance.
XAI-liacs/LLaMEA	A framework that generates and refines metaheuristic algorithms iteratively.	Medium. Good if you want to evolve a Python class that solves Max-SAT.
2. How to Build Your Own (Conceptual Architecture)
Since a specific tool doesn't exist, the standard approach in current research (like LMEA or FunSearch) is to replace standard genetic operators with LLM prompts.

The Loop:

Population: A set of candidate truth assignments (e.g., [0, 1, 1, 0, ...]).

LLM "Crossover": Prompt the LLM with two high-scoring parents. Ask it to "reason" about why they are good and combine their logic to form a new assignment.

LLM "Mutation": Prompt the LLM with a solution and the unsatisfied clauses. Ask it to flip variables specifically to fix those clauses.

Evaluation: Use a fast classical solver (like pysat) to calculate the fitness (number of satisfied clauses).

Minimal Python Prototype
You can build a basic version using openai and python-sat in about 50 lines of code.

Python
import random
from pysat.formula import CNF
from pysat.examples.rc2 import RC2 # MaxSAT solver for validation
import openai

# 1. Load MaxSAT Instance
formula = CNF(from_file='problem.cnf')

def evaluate_fitness(assignment):
    # Count satisfied clauses
    score = 0
    for clause in formula.clauses:
        if any((lit > 0 and assignment[abs(lit)-1] == 1) or 
               (lit < 0 and assignment[abs(lit)-1] == 0) for lit in clause):
            score += 1
    return score

# 2. Define LLM Mutation Operator
def llm_mutate(current_solution, unsatisfied_clauses):
    prompt = f"""
    I have a Max-SAT problem.
    Current variable assignment: {current_solution}
    These clauses are currently UNSATISFIED: {unsatisfied_clauses[:50]}...
    
    Please suggest a modified assignment that might satisfy more clauses. 
    Focus on flipping variables that appear frequently in the unsatisfied list.
    Return ONLY the new python list of bits.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    # Parse list from response...
    return new_assignment

# 3. Evolutionary Loop
population = [[random.randint(0, 1) for _ in range(formula.nv)] for _ in range(5)]

for generation in range(10):
    # Evaluate
    scored_pop = [(sol, evaluate_fitness(sol)) for sol in population]
    scored_pop.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Gen {generation} Best Score: {scored_pop[0][1]}")
    
    # Selection (Keep top 2)
    parents = [p[0] for p in scored_pop[:2]]
    
    # Reproduction (LLM Mutation on best parents)
    new_pop = parents[:] # Elitism
    while len(new_pop) < 5:
        parent = random.choice(parents)
        # Identify unsatisfied clauses for context
        # ... (logic to find bad clauses)
        child = llm_mutate(parent, bad_clauses) 
        new_pop.append(child)
    
    population = new_pop
3. Key Research Papers to Cite/Read
If you are building this for academic purposes, these are the seminal papers you should look at for methodology:

FunSearch (Nature 2023): Mathematical discoveries from program search with large language models. (Uses LLM to write code, not just mutate bits).

LMEA (ArXiv 2023): Large Language Models as Evolutionary Optimizers. (Directly proposes using LLMs as crossover/mutation operators for TSP, adaptable to Max-SAT).

Eur., et al. (2024): Copilot for SAT Control. (Investigates using LLMs to guide CDCL heuristics).

Next Step
Would you like me to generate a fully functional Python script that sets up a simple Max-SAT problem (using pysat) and uses a mock LLM function so you can test the evolutionary loop locally?

Yes

Here is a complete, runnable Python prototype.

This script sets up the entire evolutionary framework. Currently, it uses a Mock LLM function (which simulates intelligence using heuristics) so you can run the code immediately without an API key. I have marked exactly where to paste your actual OpenAI/Anthropic API call.

1. Prerequisites
You will need the python-sat library to handle the boolean logic efficiently.

Bash
pip install python-sat
2. The Python Script (llm_maxsat.py)
Python
import random
import json
from pysat.formula import CNF

# ==========================================
# 1. SETUP: Generate/Load a Max-SAT Problem
# ==========================================
def generate_random_maxsat(n_vars=20, n_clauses=80):
    """Generates a random 3-SAT problem."""
    formula = CNF()
    for _ in range(n_clauses):
        # Create a clause with 3 random literals
        clause = []
        while len(clause) < 3:
            lit = random.randint(1, n_vars)
            if random.random() < 0.5: lit = -lit
            if lit not in clause and -lit not in clause:
                clause.append(lit)
        formula.append(clause)
    return formula, n_vars

# ==========================================
# 2. CORE: Fitness & Analysis Functions
# ==========================================
def evaluate_fitness(assignment, formula):
    """
    Returns (score, bad_clauses).
    Score = number of satisfied clauses.
    Assignment is a list of 0s and 1s (index 0 = variable 1).
    """
    score = 0
    unsatisfied_indices = []
    
    for idx, clause in enumerate(formula.clauses):
        is_satisfied = False
        for lit in clause:
            var_idx = abs(lit) - 1
            val = assignment[var_idx]
            # If lit is positive and val is 1, OR lit is neg and val is 0 -> SAT
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                is_satisfied = True
                break
        
        if is_satisfied:
            score += 1
        else:
            unsatisfied_indices.append(idx)
            
    return score, unsatisfied_indices

# ==========================================
# 3. THE LLM OPERATOR (The "Brain")
# ==========================================
def llm_mutation_operator(current_assignment, formula, unsatisfied_indices):
    """
    Uses an LLM to reason about which bits to flip to fix unsatisfied clauses.
    """
    
    # A. CONSTRUCT THE PROMPT
    # We grab a sample of unsatisfied clauses to show the LLM
    sample_size = min(5, len(unsatisfied_indices))
    bad_clauses_sample = [formula.clauses[i] for i in unsatisfied_indices[:sample_size]]
    
    prompt = f"""
    You are an evolutionary operator for a SAT solver.
    Current Assignment (0 or 1 for variables 1 to {len(current_assignment)}):
    {current_assignment}
    
    The following clauses are currently UNSATISFIED (variables are 1-indexed):
    {bad_clauses_sample}
    ... (and {len(unsatisfied_indices) - sample_size} others)
    
    Your Goal: Suggest a list of variable INDICES (1-based) to FLIP (0->1 or 1->0) to satisfy these clauses.
    Reasoning: If a clause is [1, -3, 5] and is unsatisfied, it means currently 1=0, 3=1, 5=0. Flipping any one of them might help.
    
    Output JSON only: {{"flip_indices": [1, 5, ...]}}
    """
    
    # ---------------------------------------------------------
    # B. CALL THE LLM (MOCK VERSION)
    # ---------------------------------------------------------
    # INSTRUCTION: Replace the block below with actual OpenAI/Anthropic code.
    # Example: response = openai.ChatCompletion.create(...)
    
    # --- START MOCK RESPONSE ---
    # Simulating an LLM that looks at the bad clauses and picks a variable to flip.
    if bad_clauses_sample:
        # Pick a random variable from the first unsatisfied clause to flip
        # This simulates the LLM saying "Hey, flip var X to fix this clause"
        suggestion = abs(random.choice(bad_clauses_sample[0]))
        mock_response_json = {"flip_indices": [suggestion]}
    else:
        # If perfectly solved (rare), flip nothing
        mock_response_json = {"flip_indices": []}
    # --- END MOCK RESPONSE ---
    
    # C. APPLY MUTATION
    new_assignment = current_assignment[:]
    flips = mock_response_json.get("flip_indices", [])
    
    for idx in flips:
        # Convert 1-based index to 0-based
        list_idx = idx - 1
        if 0 <= list_idx < len(new_assignment):
            new_assignment[list_idx] = 1 - new_assignment[list_idx] # Bit flip
            
    return new_assignment

# ==========================================
# 4. MAIN EVOLUTIONARY LOOP
# ==========================================
def run_evolution():
    # Settings
    N_VARS = 50
    N_CLAUSES = 200
    POP_SIZE = 10
    GENERATIONS = 15
    
    # Init Problem
    formula, n_vars = generate_random_maxsat(N_VARS, N_CLAUSES)
    print(f"--- Problem Generated: {N_VARS} vars, {N_CLAUSES} clauses ---")

    # Init Population (Random assignments)
    population = [[random.randint(0, 1) for _ in range(n_vars)] for _ in range(POP_SIZE)]
    
    best_overall_score = -1
    best_solution = None

    for gen in range(GENERATIONS):
        # 1. Evaluation
        scored_pop = []
        for individual in population:
            score, bad_indices = evaluate_fitness(individual, formula)
            scored_pop.append({
                "genes": individual,
                "score": score,
                "bad_clauses": bad_indices
            })
            
            if score > best_overall_score:
                best_overall_score = score
                best_solution = individual[:]
        
        # Sort by fitness (Descending)
        scored_pop.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"Gen {gen+1}: Best Score = {scored_pop[0]['score']}/{N_CLAUSES}")
        
        if scored_pop[0]['score'] == N_CLAUSES:
            print("Found perfect solution!")
            break

        # 2. Selection (Elitism - Keep top 20%)
        cutoff = int(POP_SIZE * 0.2)
        next_gen = [ind["genes"] for ind in scored_pop[:cutoff]]
        
        # 3. LLM-Driven Reproduction
        # We fill the rest of the population by taking the best parents 
        # and asking the LLM to mutate them to fix their errors.
        while len(next_gen) < POP_SIZE:
            parent_obj = random.choice(scored_pop[:cutoff])
            parent_genes = parent_obj["genes"]
            bad_clauses = parent_obj["bad_clauses"]
            
            # THE MAGIC STEP: LLM attempts to improve the parent
            child_genes = llm_mutation_operator(parent_genes, formula, bad_clauses)
            next_gen.append(child_genes)
            
        population = next_gen

    print("\n--- End of Evolution ---")
    print(f"Best Score Found: {best_overall_score}/{N_CLAUSES}")
    print(f"Best Assignment: {best_solution}")

if __name__ == "__main__":
    run_evolution()
3. How to Connect the Real LLM
To make this "real," go to the llm_mutation_operator function and replace the Mock block with this standard OpenAI call:

Python
import openai

# ... inside llm_mutation_operator ...

response = openai.chat.completions.create(
    model="gpt-4o", # or "gpt-3.5-turbo" for speed
    messages=[
        {"role": "system", "content": "You are a Max-SAT solver assistant. Output valid JSON only."},
        {"role": "user", "content": prompt}
    ],
    response_format={"type": "json_object"} # Forces JSON output
)

content = response.choices[0].message.content
result = json.loads(content)
flips = result.get("flip_indices", [])
Why this architecture works
Context Window: We only send the LLM the unsatisfied clauses (or a sample of them). This fits a large problem into a small context window.

Targeted Mutation: Standard genetic algorithms flip bits randomly. The LLM flips bits intentionally based on the logic of the unsatisfied clause (e.g., "Clause 5 requires A or B to be true; currently both are false, so I will flip A").

Hybrid Approach: We use Python for the heavy lifting (calculating fitness) and the LLM only for the "creative" step of suggesting improvements.

Would you like me to refine the prompt to use "Chain of Thought" reasoning (asking the LLM to explain why it is flipping specific bits) to potentially improve the accuracy of the mutations?

מדיניות הפרטיות של Googleהקישור ייפתח בחלון חדש
התנאים וההגבלות של Googleהקישור ייפתח בחלון חדש
שמירה על הפרטיות שלך בממשקי Geminiהקישור ייפתח בחלון חדש
‫Gemini עלול לתת מידע לא מדויק, גם על אנשים, לכן חשוב לבדוק טוב את התשובות שלו.

