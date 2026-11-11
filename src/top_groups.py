"""Top-level bubble groups: 9 named categories people can start from.

The names are chosen up front, so this step is guided rather than fully unsupervised:
each group is described by a handful of example task statements, written in the same
style as O*NET tasks. Those sentences are embedded with the same model as the jobs, and
their average becomes the group's "prototype" vector. Each job then goes to the group
whose prototype is most similar (zero-shot classification by embedding similarity).

An optional refinement step then moves each prototype to the centre of the jobs
assigned to it and reassigns (a few rounds of K-Means seeded with the prototypes),
which lets the data sharpen the boundaries while keeping the groups' meaning.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from load_onet import PROCESSED
from text_clustering import load_embeddings

EMBED_MODEL = "all-mpnet-base-v2"

GROUPS = [
    {
        "name": "Care",
        "tagline": "I want to care for people's health and wellbeing",
        "examples": [
            "Assess patients' health problems and provide medical treatment.",
            "Administer medications and monitor patients' vital signs.",
            "Provide physical, occupational or speech therapy to help patients recover.",
            "Diagnose and treat diseases, injuries or disorders.",
            "Assist elderly, ill or disabled people with daily living activities.",
            "Provide personal care services such as bathing, grooming or massage.",
        ],
    },
    {
        "name": "Guide",
        "tagline": "I want to teach, counsel and help people grow",
        "examples": [
            "Teach students and prepare lessons, assignments and exams.",
            "Counsel individuals or families to help them resolve personal problems.",
            "Advise students on education and career plans.",
            "Train, coach or mentor people to develop their skills.",
            "Connect clients with social services and community resources.",
            "Lead religious services and provide spiritual guidance.",
        ],
    },
    {
        "name": "Protect",
        "tagline": "I want to keep people, places and rights safe",
        "examples": [
            "Patrol areas to prevent crime and enforce laws.",
            "Respond to fires and emergencies to rescue people and protect property.",
            "Investigate crimes, gather evidence and arrest suspects.",
            "Guard buildings and monitor security systems.",
            "Inspect workplaces and facilities for compliance with safety regulations.",
            "Represent clients in court and interpret laws and legal rulings.",
        ],
    },
    {
        "name": "Build",
        "tagline": "I want to build, fix and operate things with my hands",
        "examples": [
            "Construct, install and repair building structures, pipes or electrical wiring.",
            "Repair and maintain machines, engines and mechanical equipment.",
            "Set up and operate production machines to manufacture products.",
            "Operate cranes, forklifts or heavy equipment to move materials.",
            "Cut, shape and assemble wood, metal or other materials using hand and power tools.",
            "Plant, cultivate and harvest crops or care for livestock.",
        ],
    },
    {
        "name": "Invent",
        "tagline": "I want to design and engineer new technology",
        "examples": [
            "Design and develop new products, machines or systems.",
            "Write, test and debug software programs and applications.",
            "Apply engineering principles to design structures, devices or processes.",
            "Develop prototypes and test new technologies.",
            "Design computer networks, databases and information systems.",
            "Improve manufacturing processes and product designs.",
        ],
    },
    {
        "name": "Discover",
        "tagline": "I want to research, analyze and understand how things work",
        "examples": [
            "Conduct scientific research and experiments to test hypotheses.",
            "Collect and analyze data using statistical methods.",
            "Study living organisms, chemicals or physical phenomena in the laboratory.",
            "Publish research findings in scientific journals.",
            "Investigate social, economic or historical questions through research.",
            "Collect field samples and analyze environmental conditions.",
        ],
    },
    {
        "name": "Create",
        "tagline": "I want to make art, media and designs",
        "examples": [
            "Create original artwork, illustrations or designs.",
            "Write articles, stories, scripts or other content.",
            "Perform music, acting or dance for audiences.",
            "Photograph, film or edit video and audio productions.",
            "Design graphics, layouts, fashion or interior spaces.",
            "Design logos, illustrations and page layouts for print and digital media.",
            "Direct or produce films, shows or performances.",
        ],
    },
    {
        "name": "Connect",
        "tagline": "I want to serve the public, sell, and keep people and goods moving",
        "examples": [
            "Sell products or services to customers and negotiate prices.",
            "Drive buses, taxis or trains over scheduled routes to transport passengers.",
            "Drive trucks to deliver goods and cargo to customers.",
            "Pilot aircraft or ships to carry passengers and freight.",
            "Greet customers, answer questions and resolve complaints.",
            "Prepare and serve food and beverages to guests.",
            "Cook and plate meals for restaurant customers.",
            "Help shoppers in stores find, select and purchase merchandise.",
            "Promote products through advertising, marketing and public relations.",
            "Arrange travel, lodging, events or entertainment for clients.",
            "Build relationships with clients and represent the organization to the public.",
        ],
    },
    {
        "name": "Organize",
        "tagline": "I want to manage, plan and keep things running",
        "examples": [
            "Plan, direct and coordinate the operations of an organization.",
            "Prepare budgets, financial statements and accounting records.",
            "Maintain files, schedules and records and perform clerical tasks.",
            "Hire, train and supervise staff.",
            "Coordinate purchasing, inventory and the movement of goods.",
            "Process payments, claims and other business transactions.",
        ],
    },
]
GROUP_NAMES = [g["name"] for g in GROUPS]

# Fixed placements that override the similarity-based assignment.
# All transportation jobs go to Connect: O*NET's transportation occupations are SOC minor
# groups 53-2 (air), 53-3 (motor vehicle), 53-4 (rail), 53-5 (water) and 53-6 (other, e.g.
# parking attendants, traffic technicians, transportation inspectors), plus the supervisors
# of vehicle, cargo and passenger work. Material moving (53-7: cranes, packers, laborers)
# is not transportation and stays with the similarity assignment.
TRANSPORTATION_SOC_GROUPS = ("53-2", "53-3", "53-4", "53-5", "53-6")
TRANSPORTATION_SUPERVISORS = {
    "53-1041.00",  # Aircraft Cargo Handling Supervisors
    "53-1043.00",  # First-Line Supervisors of Material-Moving Machine and Vehicle Operators
    "53-1044.00",  # First-Line Supervisors of Passenger Attendants
}


# Delivery and logistics jobs also go to Connect ("keep people and goods moving").
# O*NET files most of these under office or business work, so they're listed by code.
# Not included despite similar words: 911 dispatchers and border protection officers (Protect),
# meter readers, power dispatchers, data warehousing.
DELIVERY_AND_LOGISTICS = {
    # Delivery
    "43-5021.00",  # Couriers and Messengers
    "43-5051.00",  # Postal Service Clerks
    "43-5052.00",  # Postal Service Mail Carriers
    "43-5053.00",  # Postal Service Mail Sorters, Processors, and Processing Machine Operators
    "43-9051.00",  # Mail Clerks and Mail Machine Operators, Except Postal Service
    "11-9131.00",  # Postmasters and Mail Superintendents
    # Logistics
    "13-1081.00",  # Logisticians
    "13-1081.01",  # Logistics Engineers
    "13-1081.02",  # Logistics Analysts
    "11-3071.04",  # Supply Chain Managers
    "11-3071.00",  # Transportation, Storage, and Distribution Managers
    "43-5011.00",  # Cargo and Freight Agents
    "43-5011.01",  # Freight Forwarders
    "13-1041.08",  # Customs Brokers
    "43-5071.00",  # Shipping, Receiving, and Inventory Clerks
    "43-5061.00",  # Production, Planning, and Expediting Clerks
    "43-5032.00",  # Dispatchers, Except Police, Fire, and Ambulance
}


def is_transportation(code: str) -> bool:
    return code[:4] in TRANSPORTATION_SOC_GROUPS or code in TRANSPORTATION_SUPERVISORS


def is_moving_people_or_goods(code: str) -> bool:
    """Transportation, delivery and logistics: all placed in Connect."""
    return is_transportation(code) or code in DELIVERY_AND_LOGISTICS


def apply_rules(groups: pd.Series) -> pd.Series:
    """Apply the fixed placements above to a Series of group names indexed by SOC code."""
    groups = groups.copy()
    groups[[is_moving_people_or_goods(c) for c in groups.index]] = "Connect"
    return groups


def prototypes(model_name: str = EMBED_MODEL) -> list[np.ndarray]:
    """For each group, the unit-length embeddings of its example sentences (examples x dim).

    Cached in data/processed/, keyed by a hash of the examples, so editing GROUPS
    automatically triggers a re-embed and normal builds don't need to load the model."""
    key = hashlib.md5(json.dumps([model_name, GROUPS], sort_keys=True).encode()).hexdigest()[:10]
    cache = PROCESSED / f"group_examples_{key}.npz"
    if cache.exists():
        data = np.load(cache)
        return [data[g["name"]] for g in GROUPS]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    P = [model.encode(g["examples"], normalize_embeddings=True) for g in GROUPS]
    np.savez(cache, **{g["name"]: E for g, E in zip(GROUPS, P)})
    return P


def example_scores(X: np.ndarray, P: list[np.ndarray]) -> np.ndarray:
    """jobs x groups: cosine similarity to the mean of each group's example sentences."""
    M = np.vstack([E.mean(axis=0) for E in P])
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return X @ M.T


def _unit(A: np.ndarray) -> np.ndarray:
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def assign(X: np.ndarray, P: list[np.ndarray], refine_rounds: int = 0, anchor: float = 0.7,
           features: np.ndarray | None = None) -> np.ndarray:
    """Assign each job (row of X) to its best-matching group.

    refine_rounds > 0 runs a seeded K-Means-style refinement: each round, a job's score for a
    group blends its similarity to the group's examples (weight `anchor`) with its cosine
    similarity to the centre of the jobs currently in that group. The centres are computed in
    `features` space (default: the text embeddings). Passing text + numeric features lets the
    skills profile weigh in, e.g. bus drivers' tasks mention safety a lot, but their abilities
    (reaction time, vehicle control) look like other operators.
    """
    F = _unit(X if features is None else features)
    base = example_scores(X, P)
    labels = base.argmax(axis=1)
    for _ in range(refine_rounds):
        centres = _unit(np.vstack([F[labels == g].mean(axis=0) for g in range(len(P))]))
        labels = (anchor * base + (1 - anchor) * (F @ centres.T)).argmax(axis=1)
    return labels


def top_groups(refine_rounds: int = 8, anchor: float = 0.7) -> pd.Series:
    """Group name for every job that has an embedding, with the fixed rules applied.

    anchor=0.7 was chosen by checking known jobs (see notebooks/06_top_groups.ipynb): higher values
    stop refinement from fixing cases like Chefs (Organize -> Connect) and Graphic Designers
    (Invent -> Create)."""
    X, codes = load_embeddings(EMBED_MODEL)
    labels = assign(X, prototypes(), refine_rounds=refine_rounds, anchor=anchor)
    groups = pd.Series(np.array(GROUP_NAMES)[labels], index=codes.values, name="group")
    return apply_rules(groups)
