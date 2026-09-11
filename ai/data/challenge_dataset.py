"""
Unseen Challenge Dataset Generator for CivicFlow AI (Phase 3).

Generates a realistic, unseen challenge dataset of ~400 municipal complaints with
substantial surface phrasing differences, contextual descriptions, complex Hinglish/Marathi,
and 9 explicit hard-negative confusion groups for generalization testing.
"""

import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

# Ensure workspace root in sys.path
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai.taxonomy.taxonomy_manager import TaxonomyManager


class ChallengeDatasetGenerator:
    """Generates unseen challenge dataset for robust generalization testing."""

    LOCALITIES = [
        "Aundh", "Baner", "Kothrud", "Hadapsar", "Viman Nagar", "Kalyani Nagar",
        "Wakad", "Shivajinagar", "Camp", "Magarpatta", "Bavdhan", "Katraj", "Swargate"
    ]

    # Challenging, non-repetitive contextual complaints for every category
    CHALLENGE_TEMPLATES = {
        "SAN-GARB": [
            "For the past several mornings the municipal collection van has completely skipped our alley in {locality}.",
            "Residences near {locality} main gate are forced to store rotting kitchen garbage inside houses as sanitation truck failed to arrive.",
            "Amchya {locality} society madhe kachryachi gadi regular yenar nahi, 4 divsa pasun kachra gharaat sathla aahe.",
            "Kachra van walo ne humare lane ko bypass kar diya {locality} me 3 din se."
        ],
        "SAN-DUMP": [
            "Heavy commercial dumpers dumping construction rubble and hazardous waste on open plot near {locality} school.",
            "Unauthorized dumping of chemical barrels and plastic sacks happening near {locality} bypass road.",
            "Raat ke waqt commercial trucks {locality} ke open ground par heavy waste dump kar rahe hain.",
            "Khulya jaga madhe mothya pramanat anadhikrut kachra aani waste dump kela jaat aahe {locality} madhe."
        ],
        "SAN-BIN": {
            "EN": [
                "Public dustbin metal frame corroded and broken in {locality}, all trash spilling on pedestrian walk.",
                "Community waste bin overflowing beyond capacity at {locality} market junction."
            ],
            "MR": [
                "Chowkatli kachra kundi phutli aahe aani {locality} madhe kachra bahaar ala aahe."
            ],
            "HI": [
                "Kachra kundi toot chuki hai {locality} me aur pura kachra raste par phail gaya hai."
            ]
        },
        "SAN-SWEEP": {
            "EN": [
                "Main avenue in {locality} not swept for over a week, dry leaves and thick dust causing breathing discomfort.",
                "Sanitation sweepers ignoring internal residential lanes of {locality}."
            ],
            "MR": [
                "Rasta safai kamgar {locality} madhe aala nahi, rastyavar khup mati aani kachra sachalay."
            ],
            "HI": [
                "Street sweeping nahi ho rahi hai {locality} residential sector me."
            ]
        },
        "WTR-LEAK": {
            "EN": [
                "High pressure underground supply line burst causing water gushing out onto {locality} main avenue.",
                "Purbicha water supply pipeline leak hot aahe {locality} bus stop jawad, lakhon liter pani phukat vahat aahe.",
                "Major municipal pipe leakage forming a stream near {locality} signal."
            ]
        },
        "WTR-QUAL": {
            "EN": [
                "Water coming from domestic tap in {locality} is dark brown and emitting strong foul odor.",
                "Contaminated muddy water being supplied during morning water hours in {locality}.",
                "Pinyache pani ghanerdya vasache aani mati misallele yetahe {locality} madhe."
            ]
        },
        "WTR-PRES": {
            "EN": [
                "Water pressure in municipal line in {locality} is so low that overhead tanks cannot be filled.",
                "No water supply received in {locality} sector for the third consecutive day.",
                "Pani khup kamya dabane yetahe {locality} society madhe, tyanule pani chadhat nahi."
            ]
        },
        "SEW-BLOCK": {
            "EN": [
                "Underground sewer pipe choked causing black gutter liquid to overflow into residential compounds in {locality}.",
                "Gatar tumble aahe aani ghanerde pani {locality} main rastyavar ala aahe.",
                "Sewage line backup flooding pedestrian walkway in {locality} with unbearable stench."
            ]
        },
        "SEW-MAN": {
            "EN": [
                "Cast iron manhole cover missing on unlit bend in {locality}, huge hazard for two-wheelers.",
                "Broken drain chamber slab creating open pit on footpath in {locality}.",
                "Gatarache dhakan ughade aahe {locality} rastyavar, motyabpramanat accident honyachi bhiti."
            ]
        },
        "STM-LOG": {
            "EN": [
                "Monsoon rainwater submerged railway subway underpass near {locality}, vehicles trapped.",
                "Rastyavar khup pani sachalay {locality} madhe pavsa nantar, gadyan chalat nahiye.",
                "Standing rainwater accumulation blocking residential gates in {locality}."
            ]
        },
        "STM-DRN": {
            "EN": [
                "Stormwater drain choked with plastic bags and silt in {locality} causing street flooding.",
                "Major stormwater nallah uncleaned and blocked near {locality} bridge before rains.",
                "Nalla saaf kela nahiye {locality} madhe plastic garbage mule block zala."
            ]
        },
        "RDW-POT": {
            "EN": [
                "Multiple deep craters and dangerous potholes formed on main bridge approach at {locality}.",
                "Rastyavar motha khadda padla aahe {locality} chowkat, bike slip hotat.",
                "Dangerous pothole near signal in {locality} damaging vehicle suspension."
            ]
        },
        "RDW-WALK": {
            "EN": [
                "Interlocking footpath pavers broken and missing in {locality}, elderly pedestrians falling.",
                "Footpath pavement tiles crushed and destroyed near {locality} market area.",
                "Footpath tutla aahe {locality} main road var, chaaltan khup tras hotoe."
            ]
        },
        "RDW-RD": {
            "EN": [
                "Road excavated for underground cable laying was left unpaved without asphalt in {locality}.",
                "Entire tar road surface crumbling and uneven in {locality}, urgent resurfacing needed.",
                "Rasta khadun thevla aahe {locality} madhe, damarikaran kela nahiye."
            ]
        },
        "ELE-LGT": {
            "EN": [
                "All LED streetlights along {locality} main avenue non-functional for past 5 days.",
                "Dark stretch of road in {locality} due to broken streetlight fittings creating security risk.",
                "Street lights band aahet {locality} gallyat, full andhar aahe."
            ]
        },
        "ELE-POLE": {
            "EN": [
                "Tilted electrical light pole leaning dangerously over road with open junction box in {locality}.",
                "Live electric wire hanging low near children park entrance in {locality}.",
                "Vidyut khamba kalala aahe {locality} madhe, live wire ughadi aahe."
            ]
        },
        "PRK-TREE": {
            "EN": [
                "Heavy tree branch snapped during storm blocking both traffic lanes in {locality}.",
                "Overgrown tree branches entangling with high voltage power lines in {locality}.",
                "Zadachi mothi phaandi tutun rastyavar padli aahe {locality} madhe."
            ]
        },
        "PRK-MAINT": {
            "EN": [
                "Municipal garden swings broken and rusted in {locality}, lawn unwatered and full of weeds.",
                "Baghichi dekhbhal hot nahiye {locality} garden madhe, benches tutle aahet."
            ]
        },
        "SAN-PEST": {
            "EN": [
                "Stagnant water pools breeding mosquitoes in {locality}, high dengue threat, request fogging.",
                "Machhar khup zale aahet {locality} madhe, chemical fogging gadi pathva."
            ]
        },
        "SAN-ANML": {
            "EN": [
                "Dead stray dog lying on highway shoulder in {locality}, urgent sanitary carcass removal required.",
                "Pack of aggressive stray dogs chasing pedestrians at night in {locality}."
            ]
        },
        "SAN-HYG": {
            "EN": [
                "Public toilet facility near {locality} station filthy, choked, without water connection.",
                "Sarvajanik shauchalay ghanerde aahe {locality} madhe, clean kela nahi."
            ]
        },
        "RDW-BLDG": {
            "EN": [
                "Unauthorized extra story construction underway without civic building permit in {locality}.",
                "Anadhikrut baandhkaam suru aahe {locality} madhe bina permission."
            ]
        },
        "RDW-DEBR": {
            "EN": [
                "Builder dumped heavy concrete debris and bricks on public street in {locality}.",
                "Baandhkaam malba rastyavar takla aahe {locality} madhe blocking access."
            ]
        },
        "RDW-ENC": {
            "EN": [
                "Illegal vendors and hawkers completely occupying pedestrian footpath in {locality}.",
                "Footpath var anadhikrut dukane aali aahet {locality} market madhe."
            ]
        },
        "RDW-TRF": {
            "EN": [
                "Abandoned scrap vehicle double parked blocking lane entrance in {locality}.",
                "Illegal parking of heavy commercial trucks obstructing residential gate in {locality}."
            ]
        },
        "RDW-SIGN": {
            "EN": [
                "Speed limit warning board broken and missing near school zone in {locality}.",
                "Traffic direction board damaged at {locality} intersection."
            ]
        },
        "TAX-ERR": {
            "EN": [
                "Property tax assessment notice calculated for double the square footage of property in {locality}.",
                "Ghar patti bill madhe chuk aali aahe {locality} assessment account madhe."
            ]
        },
        "TAX-REC": {
            "EN": [
                "Paid property tax online but payment receipt failed to generate for {locality} property.",
                "Tax payment status showing pending despite bank debit for {locality} account."
            ]
        },
        "LIC-TRD": {
            "EN": [
                "Unlicensed commercial garage operating in residential basement in {locality}.",
                "Bin parvangi commercial dukan suru aahe {locality} residential sector madhe."
            ]
        },
        "LIC-NOISE": {
            "EN": [
                "Event lawn playing high decibel music past 1 AM without noise permit in {locality}.",
                "Loudspeaker playing loudly beyond permitted hours near hospital in {locality}."
            ]
        },
        "PRK-AIR": {
            "EN": [
                "Open burning of plastic garbage in vacant plot at {locality} causing toxic smog.",
                "Kachra aani dry leaves jalale jaat aahet {locality} madhe causing heavy smoke."
            ]
        },
        "PRK-NOIS": {
            "EN": [
                "Heavy industrial generator running continuous night shifts causing severe noise in {locality}.",
                "Unbearable machine noise pollution in {locality} quiet residential zone."
            ]
        },
        "SAF-FIRE": {
            "EN": [
                "Fire exit and emergency staircase blocked with wooden crates in {locality} building.",
                "Defective fire safety extinguishers in {locality} commercial complex."
            ]
        },
        "SAF-DIS": {
            "EN": [
                "Old building wall cracked severely and leaning precariously over street in {locality}.",
                "Imminent structural collapse hazard of dilapidated wall in {locality}."
            ]
        }
    }

    HARD_NEGATIVE_GROUPS = {
        "GROUP_1_WASTE": ["SAN-GARB", "SAN-DUMP", "SAN-BIN"],
        "GROUP_2_WATER": ["WTR-LEAK", "WTR-QUAL"],
        "GROUP_3_WATER_SEWER": ["WTR-LEAK", "SEW-BLOCK"],
        "GROUP_4_ROADS": ["RDW-POT", "RDW-WALK", "RDW-RD"],
        "GROUP_5_STORM_SEWER": ["STM-LOG", "SEW-BLOCK"],
        "GROUP_6_LIGHT_FIRE": ["ELE-POLE", "SAF-FIRE"],
        "GROUP_7_HEALTH_ENV": ["SAN-PEST", "PRK-AIR"],
        "GROUP_8_BUILDING_ENC": ["RDW-BLDG", "RDW-ENC"],
        "GROUP_9_TRAFFIC_ROADS": ["RDW-TRF", "RDW-RD"]
    }

    def __init__(self, taxonomy_manager: TaxonomyManager = None):
        self.tm = taxonomy_manager or TaxonomyManager()

    def generate_challenge_dataset(self, target_per_category: int = 12, seed: int = 123) -> List[Dict[str, Any]]:
        """Generates ~400 unseen challenge records across all 34 categories."""
        random.seed(seed)
        all_categories = self.tm.get_categories()
        dataset = []
        doc_id = 9000

        for cat_code, cat_info in all_categories.items():
            # Get domain
            domain_code = None
            for d_code, d_data in self.tm.data.get("domains", {}).items():
                if cat_code in d_data.get("categories", {}):
                    domain_code = d_code
                    break

            dept_code = cat_info.get("department_mapping", {}).get("department_code")
            tpls = self.CHALLENGE_TEMPLATES.get(cat_code, {})

            if isinstance(tpls, dict):
                flat_list = []
                for v in tpls.values():
                    flat_list.extend(v)
                tpls = flat_list

            if not tpls:
                tpls = [f"Unseen complaint regarding {cat_info.get('category_name')} in {{locality}}."]

            for i in range(target_per_category):
                doc_id += 1
                loc = random.choice(self.LOCALITIES)
                t_str = random.choice(tpls)
                text = t_str.format(locality=loc) if "{locality}" in t_str else t_str

                # Language detection
                lang = "EN"
                if any(w in text.lower() for w in ["madhe", "aahe", "zala", "rastyavar", "gatyar", "kachra"]):
                    lang = "HINGLISH" if not any(ord(c) > 127 for c in text) else "MARATHI"

                dataset.append({
                    "id": f"CHALLENGE-{doc_id}",
                    "complaint_text": text,
                    "domain_code": domain_code,
                    "category_code": cat_code,
                    "department_code": dept_code,
                    "language": lang,
                    "difficulty": "CHALLENGE_UNSEEN",
                    "source": "unseen_challenge_dataset"
                })

        return dataset


if __name__ == "__main__":
    gen = ChallengeDatasetGenerator()
    ds = gen.generate_challenge_dataset(target_per_category=12)

    processed_dir = Path(__file__).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_csv = processed_dir / "challenge_test.csv"
    fields = ["id", "complaint_text", "domain_code", "category_code", "department_code", "language", "difficulty", "source"]

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ds)

    # Save hard negatives definition
    with open(processed_dir / "hard_negatives.json", "w", encoding="utf-8") as f:
        json.dump(gen.HARD_NEGATIVE_GROUPS, f, indent=2)

    print(f"Generated {len(ds)} unseen challenge records in {out_csv}")
    print(f"Saved 9 hard negative confusion groups to {processed_dir / 'hard_negatives.json'}")
