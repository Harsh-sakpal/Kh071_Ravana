"""
Synthetic Municipal Complaint Dataset Generator for CivicFlow AI.
Dynamically reads categories from taxonomy.json and generates ~100 realistic, diverse complaints per category.
Supports English (~50%), Marathi (~25%), and Hinglish (~25%).
"""

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


class MunicipalDatasetGenerator:
    """Generates synthetic, realistic municipal complaint datasets based dynamically on taxonomy.json."""

    PUNE_LOCALITIES = [
        "Kothrud", "Baner", "Viman Nagar", "Hadapsar", "Shivajinagar", "Camp",
        "Katraj", "Wakad", "Aundh", "Kalyani Nagar", "Pimpri", "Chinchwad",
        "Swargate", "Warje", "Deccan Gymkhana", "Magarpatta", "Bavdhan", "Pashan",
        "Kondhwa", "Koregaon Park", "Bibwewadi", "Yerwada", "Sinhagad Road"
    ]

    DURATIONS_EN = ["for the last 3 days", "since yesterday", "for past 2 weeks", "since Monday", "for over 5 days"]
    DURATIONS_MR = ["maagchya 3 divsa pasun", "kaal pasun", "2 aathavde zale", "somvar pasun", "khup divsa pasun"]
    DURATIONS_HI = ["pichle 3 din se", "kal se", "2 hafte se", "kuch dino se", "lagatar"]

    SEVERITY_EN = ["Urgent attention required.", "Posing dangerous safety hazard.", "Severely affecting residents.", "Needs immediate action."]
    SEVERITY_MR = ["Khup mothi samsya aahe.", "Urgent action ghyaychi garaj aahe.", "Lokana khup tras hoto aahe.", "Kaalji ghyavi."]
    SEVERITY_HI = ["Bahut badi dikkat hai.", "Jaldi action lo.", "Logon ko bahut dikkat ho rahi hai.", "Kripya dhyan dein."]

    # Dynamic templates per category code
    CATEGORY_PATTERNS = {
        "SAN-GARB": {
            "EN": [
                "Garbage vehicle has not arrived {duration} in {locality}. Trash is piling up outside buildings.",
                "Door to door waste collection missed in {locality} {duration}. Garbage rotting near residential gate.",
                "Household waste is not collected by municipal truck in {locality}. Bad odor spreading everywhere.",
                "Sanitation van did not visit our street in {locality} {duration}. Request immediate garbage truck dispatch."
            ],
            "MR": [
                "Amchya {locality} gallyat kachra gadi {duration} aali nahiye. Kachra gharabaher sathla aahe.",
                "Gharoghari kachra sankalan gadi {locality} madhe aali nahi. Badha kachra rastyavar aala aahe.",
                "Kachra truck {duration} aala nahi, khup durdhasha zali aahe {locality} madhe.",
                "Gharogharun kachra uchalnyachi gadi {locality} bhagat aali nahi, durgandhi pasarli aahe."
            ],
            "HI": [
                "Kachra gaadi {locality} me {duration} nahi aayi hai. Kachra Society ke bahar jama ho gaya hai.",
                "Ghar se kachra uthane wali gaadi {locality} me nahi aayi. Badbu aa rahi hai.",
                "Waste collection truck {locality} me {duration} se missing hai. Kripya kachra gadi bhejein.",
                "Kachra van {locality} me nahi aayi, kachra raste par phail raha hai."
            ]
        },
        "SAN-DUMP": {
            "EN": [
                "Illegal dumping of industrial and commercial waste by trucks in open plot at {locality}.",
                "Unknown commercial vehicles dumping heaps of garbage at roadside near {locality} open ground.",
                "Unauthorized waste dumping happening at midnight in vacant land in {locality}.",
                "Bulk waste and plastic garbage dumped illegally on main road corner in {locality}."
            ],
            "MR": [
                "Khulya jagat anadhikrut kachra dumping hot aahe {locality} madhe.",
                "Truck valya lokanit {locality} rastyachya kadela kachra takla aahe.",
                "Illegal dumping hotoe {locality} jawal, commercial waste aani plastic takle aahe.",
                "Raatri che veli {locality} madhe illegal dumping kela jaat aahe."
            ],
            "HI": [
                "Khuli jagah par illegal dumping ho raha hai {locality} me.",
                "Truck dwara kachra aur plastic waste {locality} open plot me phenka ja raha hai.",
                "Roadside par illegal waste dump kar rahe hain log {locality} near market.",
                "Bina permission ke commercial dumping ho rahi hai {locality} me."
            ]
        },
        "SAN-BIN": {
            "EN": [
                "Public dustbin broken and overflowing at {locality} market corner.",
                "Community waste bin in {locality} is completely damaged and trash is spilling out.",
                "Metal garbage container broken and missing cover in {locality}, needs replacement.",
                "Overflowing public dustbin spreading filth on sidewalk at {locality}."
            ],
            "MR": [
                "Sarvajanik kachra kundi tutli aahe aani {locality} madhe kachra bahaar ala aahe.",
                "Kachra kundi phutli aahe {locality} chowk jawad, navin kundi basva.",
                "Dustbin madhun kachra rastyavar sandat aahe {locality} madhe.",
                "Public kachra bin damaged zala aahe {locality} market jawad."
            ],
            "HI": [
                "Public dustbin tut gaya hai {locality} me aur kachra bahar phail raha hai.",
                "Kachra kundi overflowing ho chuki hai {locality} corner par.",
                "Community bin completely damaged hai {locality} near station.",
                "Public kachra dabba tuta hua hai {locality} me, badlo isse."
            ]
        },
        "SAN-SWEEP": {
            "EN": [
                "Street sweeper has not swept main road in {locality} {duration}. Heavy dust and leaves.",
                "Road sweeping and cleaning completely neglected in {locality} residential lanes.",
                "Municipal sweeper not cleaning the street near {locality} park entrance.",
                "Dust and litter accumulating on roadside in {locality} due to lack of daily sweeping."
            ],
            "MR": [
                "Rasta safai karamchari {locality} madhe {duration} aala nahi, dhur aani pane sathli aahet.",
                "Galli safai zali nahiye {locality} madhe khup divsa pasun.",
                "Road sweeping hot nahiye {locality} chowkat, rastyavar khup mati aahe.",
                "Safai kamgar {locality} madhe rasta saaf karat nahiye."
            ],
            "HI": [
                "Street sweeper {locality} me {duration} se nahi aaya hai. Raste par mati aur kachra hai.",
                "Road safai nahi ho rahi hai {locality} me. Safai karamchari ko bhejo.",
                "Gali ki safai band hai {locality} me. Raste par dhool aur kachra hai.",
                "Daily street cleaning miss ho rahi hai {locality} area me."
            ]
        },
        "WTR-LEAK": {
            "EN": [
                "Main drinking water pipeline burst in {locality}. High pressure water wasting on road.",
                "Water main leakage flowing continuously under road in {locality} near bus stop.",
                "Municipal supply pipe burst in {locality}, thousands of liters of clean water wasted.",
                "Underground pipeline leaking heavily on main street of {locality}."
            ],
            "MR": [
                "Pinyachya panyachi main pipeline burst zali aahe {locality} madhe.",
                "Pani purvotha line leak aaha {locality} chowk jawad, pani phukat jaatahe.",
                "Panyachi pipeline phutli aahe {locality} madhe, rastyavar pani vahat aahe.",
                "Drinking water supply line heavy leak hot aahe {locality} near station."
            ],
            "HI": [
                "Main water pipeline burst ho gayi hai {locality} me. Paani raste par beh raha hai.",
                "Pinyache paani ki line leak hai {locality} near bus stop.",
                "Municipal water main pipe phat gaya hai {locality} me, paani waste ho raha hai.",
                "Water supply line burst near {locality} main road."
            ]
        },
        "WTR-QUAL": {
            "EN": [
                "Tap water supply is contaminated, muddy, and smells like sewage in {locality}.",
                "Blackish and foul smelling water coming from municipal tap in {locality}.",
                "Dirty water with sediment receiving during water supply hours in {locality}.",
                "Drinking water quality is contaminated in {locality}, health hazard to residents."
            ],
            "MR": [
                "Nala tun ghanerde aani ghan vasache pani yetahe {locality} madhe.",
                "Pinyache pani mati misallle aale aahe {locality} madhe, aarogya dhoka.",
                "Pani khup ghanerd aale aahe {locality} society madhe {duration}.",
                "Nalache pani kalpat aani durgandhyukta aale aahe {locality} madhe."
            ],
            "HI": [
                "Nal me ganda aur badbu dar paani aa raha hai {locality} me.",
                "Drinking water contaminated hai {locality} block C me.",
                "Pinyaka paani me gatar ka paani mix ho gaya hai {locality} me.",
                "Muddy and foul smelling tap water receiving in {locality}."
            ]
        },
        "WTR-PRES": {
            "EN": [
                "No water supply received in {locality} during morning hours {duration}.",
                "Very low water pressure in municipal pipeline in {locality}, upper floors getting no water.",
                "Total absence of drinking water supply in {locality} block for 2 days.",
                "Water supply pressure is extremely low in {locality}, tanks remain empty."
            ],
            "MR": [
                "Panyacha dab khup kamya aahe {locality} madhe, pani var chadhat nahi.",
                "Pani aalech nahiye {locality} madhe {duration}, khup halat zali aahe.",
                "Municipal water pressure kamya zala aahe {locality} society madhe.",
                "Water supply band aahe {locality} madhe sakalchya velis."
            ],
            "HI": [
                "Water supply pressure bahut kam hai {locality} me, paani tank me nahi chad raha.",
                "Paani bilkul nahi aaya {locality} me {duration} se.",
                "Low water pressure problem in {locality} residential area.",
                "Morning hours me paani nahi aa raha hai {locality} me."
            ]
        },
        "SEW-BLOCK": {
            "EN": [
                "Sewer line blocked and sewage water overflowing onto main street in {locality}.",
                "Underground drainage blocked, foul gutter water entering house compound in {locality}.",
                "Sewage line backup causing foul liquid to flood pedestrian path at {locality}.",
                "Blocked municipal sewer pipe spilling contaminated black water in {locality}."
            ],
            "MR": [
                "Gatar तुंबले aahe aani ghanerde pani rastyavar ala aahe {locality} madhe.",
                "Sewer pipeline block zali aahe {locality} chowk jawad, gatar saaf kara.",
                "Gatarache pani ghara compound madhe ghusat aahe {locality} madhe.",
                "Underground drainage clogged zala aahe {locality} main street var."
            ],
            "HI": [
                "Gatar block ho gaya hai aur badbudar paani raste par aa gaya hai {locality} me.",
                "Sewage line overflow ho rahi hai {locality} near market.",
                "Drainage pipeline chok up ho gayi hai {locality} me, gunda paani phail raha hai.",
                "Blocked sewer line causing flooding of gutter water in {locality}."
            ]
        },
        "SEW-MAN": {
            "EN": [
                "Manhole cover open and missing on busy road in {locality}. Severe accident hazard.",
                "Broken drain chamber cover creating open pit on footpath in {locality}.",
                "Gutter cover stolen/missing near school in {locality}, pedestrians falling in dark.",
                "Damaged manhole slab collapsed in {locality}, urgent repair required."
            ],
            "MR": [
                "Gatarache dhakan ughade aahe {locality} rastyavar, accident honyachi bhiti.",
                "Chamber cover tutla aahe {locality} footpath var, manhole open aahe.",
                "Open manhole cover in {locality} near bus stop, ratri lok padtat.",
                "Gatar slab tutun motha khadda zala aahe {locality} madhe."
            ],
            "HI": [
                "Gatar ka dhakkan khula hai {locality} main road par, accident ho sakta hai.",
                "Open manhole cover in {locality} near station, bahut dangerous hai.",
                "Broken chamber cover on walk path at {locality}.",
                "Manhole slab phat gaya hai {locality} me, dhakkan lagao."
            ]
        },
        "STM-LOG": {
            "EN": [
                "Severe rainwater waterlogging under subway/bridge at {locality}. Traffic blocked.",
                "Knee deep water stagnation on main road in {locality} after rain.",
                "Rainwater flooded whole road in {locality}, vehicles stuck in standing water.",
                "Monsoon waterlogging on street in {locality} due to lack of drainage outlet."
            ],
            "MR": [
                "Rastyavar khup pani sathle aahe {locality} madhe pavsa nantar.",
                "Ghatkopar subway jawal waterlogging zali aahe {locality} madhe, gadyan chalat nahi.",
                "Pavsache pani rastyavar sachalay {locality} society bahaar.",
                "Knee deep pani sathle aahe {locality} main road var."
            ],
            "HI": [
                "Road par paani bhar gaya hai {locality} me rainy season me.",
                "Waterlogging problem under flyover at {locality}, traffic jam ho raha hai.",
                "Raste par bahut paani jam gaya hai {locality} near school.",
                "Severe standing rainwater accumulation in {locality} lanes."
            ]
        },
        "STM-DRN": {
            "EN": [
                "Stormwater drain choked with plastic waste and silt in {locality} before monsoon.",
                "Roadside storm drain blocked causing rainwater backflow in {locality}.",
                "Major nallah blocked with dumped trash near {locality} bridge.",
                "Drainage channel uncleaned and clogged in {locality}, monsoon clearance needed."
            ],
            "MR": [
                "Storm water drain kachrayane block zala aahe {locality} madhe.",
                "Nalla saaf kela nahiye {locality} madhe, pavsat pani saachnar.",
                "Rastyavarchi nali block aahe plastic bags mule {locality} madhe.",
                "Stormwater channel clogged zali aahe {locality} near market."
            ],
            "HI": [
                "Storm drain plastic aur mitti se block hai {locality} me.",
                "Nalla safai nahi hui hai {locality} near bridge, drainage choked hai.",
                "Rainwater drain blockage in {locality} causing water logging risk.",
                "Roadside drain choked with rubbish in {locality}."
            ]
        },
        "RDW-POT": {
            "EN": [
                "Huge dangerous pothole on main avenue in {locality} causing bike accidents.",
                "Multiple deep craters and potholes formed on flyover approach at {locality}.",
                "Road surface broken with large pothole near signal in {locality}.",
                "Crumbling asphalt created dangerous potholes on busy lane in {locality}."
            ],
            "MR": [
                "Rastyavar motha khadda padla aahe {locality} chowkat, gadi slip hote.",
                "Khadde padle aahet {locality} main road var, rasta kharab zala aahe.",
                "Dangerous pothole near signal in {locality}, accident hoto.",
                "Pothole patching garajeshche aahe {locality} flyover jawad."
            ],
            "HI": [
                "Main road par bada pothole ban gaya hai {locality} me.",
                "Bahut sare khadde hain {locality} road par, bike accident ho rahe hain.",
                "Dangerous pothole near {locality} bus stop, patch work karo.",
                "Road par deep crater ban gaya hai {locality} me."
            ]
        },
        "RDW-WALK": {
            "EN": [
                "Footpath paving blocks broken and missing in {locality}, senior citizens tripping.",
                "Broken sidewalk tiles and uneven pavement on main road in {locality}.",
                "Pedestrian walkway completely damaged and excavated in {locality}.",
                "Curb stones loose and broken on footpath near {locality} market."
            ],
            "MR": [
                "Footpath tutla aahe {locality} madhe, lokanit chaaltan padtat.",
                "Footpath tiles phutlya aahet {locality} main avenue var.",
                "Walkway paver blocks missing aahet {locality} near station.",
                "Pedestrian footpath kharab zala aahe {locality} madhe."
            ],
            "HI": [
                "Footpath tut gaya hai {locality} me, pedestrians ko chalne me dikkat.",
                "Sidewalk pavers broken hain {locality} main road par.",
                "Broken footpath tiles in {locality} near market area.",
                "Pedestrian walk path completely broken in {locality}."
            ]
        },
        "RDW-RD": {
            "EN": [
                "Road excavated for utility trench was left unpaved and abandoned in {locality}.",
                "Entire road surface crumbled and uneven in {locality}, resurfacing needed.",
                "Dug up road left open without asphalt layer in {locality} for months.",
                "Broken road divider and damaged road surface on avenue in {locality}."
            ],
            "MR": [
                "Rasta khadun thevla aahe {locality} madhe, damar kela nahiye.",
                "Road resurfacing zale nahiye {locality} madhe trenching nantar.",
                "Rasta kharab zala aahe {locality} avenue var, damarikaran kara.",
                "Excavated road abandoned in {locality} without proper tar layer."
            ],
            "HI": [
                "Road khod ke chod diya gaya hai {locality} me, repair nahi hua.",
                "Uneven and damaged road surface in {locality} main lane.",
                "Road resurfacing required in {locality} post pipeline work.",
                "Dug up road left unpaved in {locality} area."
            ]
        },
        "ELE-LGT": {
            "EN": [
                "All streetlights on main colony street off in {locality} {duration}. Complete darkness.",
                "Streetlight outage at {locality} junction causing safety risk at night.",
                "Non-functioning LED street lamps on main road of {locality}.",
                "Dark stretch on highway near {locality} due to broken streetlight fittings."
            ],
            "MR": [
                "Rastyavarchi street light chalat nahiye {locality} madhe {duration}.",
                "Andhar aahe {locality} gallyat, street lights band aahet.",
                "LED street light phutli/band aahe {locality} chowk jawad.",
                "Street lamps non-functional aahet {locality} main road var."
            ],
            "HI": [
                "Street lights band hain {locality} me {duration} se, andhera hai.",
                "Streetlight outage on main road in {locality} near park.",
                "Light pole bulb not working in {locality} colony street.",
                "Street lamps dark in {locality} area at night."
            ]
        },
        "ELE-POLE": {
            "EN": [
                "Live electrical wire hanging dangerously low near park gate in {locality}.",
                "Tilted electrical pole with open junction box exposing wires in {locality}.",
                "Sparking open wire hanging from light pole at {locality} market.",
                "Damaged street light pole leaning towards road in {locality}."
            ],
            "MR": [
                "Vidyut khamba kalala aahe {locality} madhe, live wire ughadi aahe.",
                "Live electrical wire thongli aahe {locality} bachat park jawad.",
                "Open junction box at light pole base in {locality}, shock hazard.",
                "Light pole tilted and dangerous in {locality} main street."
            ],
            "HI": [
                "Live wire khule me latak raha hai {locality} near school.",
                "Electric pole jhuka hua hai {locality} me, wires open hain.",
                "Sparking in open wire on street light pole in {locality}.",
                "Dangerous hanging wire hazard in {locality} area."
            ]
        },
        "PRK-TREE": {
            "EN": [
                "Large tree branch fell across main road in {locality} blocking traffic.",
                "Overgrown dangerous tree branches touching electrical lines in {locality}.",
                "Dead tree leaning precariously over house roof in {locality}.",
                "Fallen tree obstruction on street in {locality} after heavy winds."
            ],
            "MR": [
                "Zadachi mothi phaandi tutun rastyavar padli aahe {locality} madhe.",
                "Overgrown tree branch electrical wire la sparsat aahe {locality} madhe.",
                "Zad padle aahe {locality} road var, rasta block zala.",
                "Dangerous leaning tree in {locality} garden boundary."
            ],
            "HI": [
                "Ped ki badi daal tut kar raste par gir gayi hai {locality} me.",
                "Tree branch overhead electric wire se touch ho rahi hai {locality} me.",
                "Fallen tree blocking street in {locality} near station.",
                "Dangerous overgrown branches trimming needed in {locality}."
            ]
        },
        "PRK-MAINT": {
            "EN": [
                "Public park play equipment broken and rusted in {locality} garden.",
                "Municipal garden unmaintained, overgrown weeds and broken boundary fence in {locality}.",
                "Park lights broken and benches damaged in {locality} public garden.",
                "Neglected green space and unwatered lawn in {locality} municipal park."
            ],
            "MR": [
                "Baghichi dekhbhal hot nahiye {locality} garden madhe, swings tutle aahet.",
                "Park benches aani lights phutle aahet {locality} municipal park madhe.",
                "Garden maintenance kharab zali aahe {locality} area madhe.",
                "Children garden swings broken in {locality} park."
            ],
            "HI": [
                "Public park ka maintenance nahi ho raha hai {locality} me.",
                "Garden swings and benches broken in {locality} municipal garden.",
                "Park lights tuti hui hain {locality} me, dekhbhal karo.",
                "Unmaintained park and broken fences in {locality}."
            ]
        },
        "SAN-PEST": {
            "EN": [
                "Severe mosquito breeding in stagnant water in {locality}, request fogging.",
                "High mosquito infestation causing dengue risk in {locality}, chemical spray needed.",
                "Vector control fogging required immediately in {locality} residential lanes.",
                "Mosquito menace in {locality} near open drain, request anti-larval treatment."
            ],
            "MR": [
                "Machhar khup zale aahet {locality} madhe, fogging gadi पाठवा.",
                "Chalktelya panyat machhar breeding hot aahe {locality} madhe.",
                "Anti mosquito fogging spray garajeshche aahe {locality} society madhe.",
                "Machharacha tras khup vadhla aahe {locality} bhagat."
            ],
            "HI": [
                "Machhar bahut zyada ho gaye hain {locality} me, fogging spray karvao.",
                "Mosquito breeding in stagnant water pools in {locality}.",
                "Request anti-mosquito fogging drive in {locality} area.",
                "Machhar menace in {locality} residential sector."
            ]
        },
        "SAN-ANML": {
            "EN": [
                "Dead stray dog lying on roadside in {locality}, sanitary removal needed.",
                "Aggressive stray dogs chasing two-wheelers at night in {locality}.",
                "Animal carcass lying on main avenue in {locality} causing stench.",
                "Stray cattle creating traffic hazard on main road in {locality}."
            ],
            "MR": [
                "Mulela prani rastyavar padla aahe {locality} madhe, saaf kara.",
                "Bhatkyan kutyancha tras vadhla aahe {locality} chowkat.",
                "Dead animal removal required on street in {locality}.",
                "Dog menace and aggressive stray pack in {locality} market."
            ],
            "HI": [
                "Mura hua janwar raste par pada hai {locality} me, usse hatvao.",
                "Stray dogs terror in {locality} colony at night.",
                "Dead animal carcass removal urgent in {locality} main lane.",
                "Stray animal menace on main road at {locality}."
            ]
        },
        "SAN-HYG": {
            "EN": [
                "Public toilet in {locality} is filthy, clogged, and without water supply.",
                "Unhygienic community restroom near station in {locality} stink spreading.",
                "Municipal public toilet dirty, uncleaned for weeks in {locality}.",
                "Community toilet block choked and dirty in {locality} market."
            ],
            "MR": [
                "Sarvajanik shauchalay ghanerde aahe {locality} madhe, pani nahiye.",
                "Public toilet safai zali nahiye {locality} station jawad.",
                "Municipal toilet block clogged aahe {locality} chowk madhe.",
                "Toilet hygine issue in {locality} public restroom."
            ],
            "HI": [
                "Public toilet bahut ganda hai {locality} me, paani bhi nahi hai.",
                "Community toilet clogged and dirty in {locality} near station.",
                "Public restroom unhygienic condition in {locality} market.",
                "Municipal toilet cleaning missed in {locality} area."
            ]
        },
        "RDW-BLDG": {
            "EN": [
                "Unauthorized extra floor construction without municipal permit in {locality}.",
                "Unsafe illegal structural alteration of building in {locality} without approval.",
                "Building construction work proceeding without safety nets or permits in {locality}.",
                "Illegal commercial extension over setback line in {locality}."
            ],
            "MR": [
                "Anadhikrut baandhkaam suru aahe {locality} madhe bina permission.",
                "Illegal construction hot aahe {locality} society bahaar.",
                "Unsafe scaffolding over public walk path in {locality}.",
                "Bina parvangi floor build kela jaat aahe {locality} madhe."
            ],
            "HI": [
                "Unauthorized construction ho raha hai {locality} me without permit.",
                "Illegal floor construction in residential building at {locality}.",
                "Unsafe scaffolding without safety net in {locality}.",
                "Bina approval illegal building modification in {locality}."
            ]
        },
        "RDW-DEBR": {
            "EN": [
                "Builder dumped construction debris and concrete waste on public lane in {locality}.",
                "Heavy debris, gravel, and sand blocking pedestrian passage in {locality}.",
                "Illegal dumping of building demolition rubble on road in {locality}.",
                "Debris left on public street post building work in {locality}."
            ],
            "MR": [
                "Baandhkaam malba aani matil rastyavar takla aahe {locality} madhe.",
                "Debris rastyavar taklyamule rasta block zala aahe {locality} madhe.",
                "Building demolition rubble dumped on footpath at {locality}.",
                "Malba removing garajeshche aahe {locality} main avenue var."
            ],
            "HI": [
                "Builder ne construction debris raste par phenka hai {locality} me.",
                "Malba blocking road in {locality} near construction site.",
                "Building material and debris left on public road at {locality}.",
                "Demolition waste dumped illegally in {locality} lanes."
            ]
        },
        "RDW-ENC": {
            "EN": [
                "Illegal hawkers and stalls completely encroaching footpath in {locality}.",
                "Shopkeeper erected permanent wooden structure covering sidewalk in {locality}.",
                "Encroachment on public road by illegal sheds and food stalls in {locality}.",
                "Commercial obstruction of pedestrian walkway by illegal vendors in {locality}."
            ],
            "MR": [
                "Footpath var anadhikrut dukane aani taprya aalya aahet {locality} madhe.",
                "Shopkeeper ne footpath encroach kela aahe {locality} market madhe.",
                "Encroachment zale aahe main street var {locality} chowkat.",
                "Footpath encroachment hawkers mule {locality} madhe."
            ],
            "HI": [
                "Footpath par illegal hawkers ne encroachment kar liya hai {locality} me.",
                "Dukandar ne footpath daba liya hai {locality} main road par.",
                "Illegal stalls blocking pedestrian path in {locality} market.",
                "Roadside encroachment by food vendors in {locality}."
            ]
        },
        "RDW-TRF": {
            "EN": [
                "Abandoned rusted vehicle illegally parked blocking residential gate in {locality}.",
                "Illegal double parking of commercial trucks blocking traffic flow in {locality}.",
                "No-parking zone violated constantly by double parked cars in {locality}.",
                "Road obstruction caused by long-term abandoned scrap car in {locality}."
            ],
            "MR": [
                "Gadi anadhikrut pane park keli aahe {locality} lane entrance var.",
                "Scrap car block karat aahe rasta {locality} society bahaar.",
                "Illegal parking mule traffic jam hotoe {locality} chowk madhe.",
                "Double parking obstruction in {locality} main market."
            ],
            "HI": [
                "Unclaimed old car raste par park karke block kar diya hai {locality} me.",
                "Illegal double parking blocking road in {locality} near market.",
                "Abandoned scrap vehicle in {locality} lane.",
                "No parking area me heavy vehicles parked in {locality}."
            ]
        },
        "RDW-SIGN": {
            "EN": [
                "Traffic direction sign post broken and fallen down at {locality} junction.",
                "Missing speed bump warning sign near school in {locality}.",
                "Damaged speed limit traffic board in {locality} needs replacement.",
                "Reflective direction sign missing at dangerous curve in {locality}."
            ],
            "MR": [
                "Traffic sign board phutla/tutla aahe {locality} chowkat.",
                "Speed breaker sign missing aahe {locality} school jawad.",
                "Rasta darsak board damaged zala aahe {locality} main road var.",
                "Traffic direction board missing in {locality} area."
            ],
            "HI": [
                "Traffic sign board tut gaya hai {locality} junction par.",
                "Speed breaker sign missing hai {locality} me near school.",
                "Damaged traffic direction board in {locality} area.",
                "Missing road sign indicator at {locality} curve."
            ]
        },
        "TAX-ERR": {
            "EN": [
                "Property tax assessment bill computed incorrectly with wrong square footage in {locality}.",
                "Received duplicate property tax invoice with incorrect penalty calculation.",
                "Property tax assessment error in civic portal bill for property in {locality}.",
                "Wrong tax slab applied on residential property tax bill in {locality}."
            ],
            "MR": [
                "Ghar patti bill madhe chuk aali aahe {locality} property sathi.",
                "Tax bill double aale aahe wrongly calculate houn {locality} madhe.",
                "Property tax assessment error aala aahe online bill madhe.",
                "Incorrect penalty applied on property tax notice in {locality}."
            ],
            "HI": [
                "Property tax bill galat calculate hua hai {locality} property ka.",
                "Wrong tax bill received with excess penalty in {locality}.",
                "Property tax assessment mistake in municipal bill for {locality}.",
                "Incorrect square feet taken for property tax in {locality}."
            ]
        },
        "TAX-REC": {
            "EN": [
                "Property tax paid online but official payment receipt not generated for {locality} property.",
                "Tax payment status showing pending despite money debited from bank for {locality}.",
                "Unable to download property tax NOC / receipt from municipal portal.",
                "Property tax payment record not updated in system for {locality} assessment."
            ],
            "MR": [
                "Property tax online bhorla pan pavti milali nahi {locality} property chi.",
                "Tax receipt generate zali nahiye online payment nantar.",
                "Payment status updating hot nahiye tax portal var.",
                "Tax pavti record pending dakhavtat {locality} account madhe."
            ],
            "HI": [
                "Property tax pay kar diya online lekin receipt nahi mili {locality} ka.",
                "Tax payment receipt not generated after online transaction for {locality}.",
                "System me tax payment record update nahi hua hai {locality} ka.",
                "Property tax receipt download error on portal."
            ]
        },
        "LIC-TRD": {
            "EN": [
                "Commercial workshop operating illegally without valid municipal trade license in {locality}.",
                "Unlicensed commercial business running in residential basement in {locality}.",
                "Shop operating in violation of trade license terms in {locality} residential area.",
                "Commercial establishment running without health and safety permit in {locality}."
            ],
            "MR": [
                "Bin parvangi commercial dukan suru aahe {locality} madhe.",
                "Trade license nastaana workshop chalavat aahet {locality} residential madhe.",
                "Unlicensed business in residential area in {locality}.",
                "Municipal license violation in {locality} commercial shop."
            ],
            "HI": [
                "Bina trade license ke commercial workshop chal raha hai {locality} me.",
                "Unlicensed shop running in residential sector of {locality}.",
                "Trade permit violation by commercial store in {locality}.",
                "Shop running without municipal license in {locality}."
            ]
        },
        "LIC-NOISE": {
            "EN": [
                "Commercial event hall playing loud music past midnight without sound permission in {locality}.",
                "High decibel loudspeaker operating beyond allowed hours near hospital in {locality}.",
                "Noise permission violation by outdoor event playing loud bass late night in {locality}.",
                "Loudspeaker playing loudly without municipal noise permit in {locality}."
            ],
            "MR": [
                "Ratri late night loud mic/speaker suru aahe {locality} madhe bina parvangi.",
                "Sound permission deadline violate zali aahe {locality} hall madhe.",
                "Aavajache pradushan khup mothya aavajat speaker chalat aahe {locality} madhe.",
                "Loudspeaker noise violation past midnight in {locality}."
            ],
            "HI": [
                "Late night loudspeaker chal raha hai {locality} me bina permission ke.",
                "Event hall violating noise permission timing in {locality}.",
                "Loud music beyond allowed hours near hospital in {locality}.",
                "Sound permit violation by party venue in {locality}."
            ]
        },
        "PRK-AIR": {
            "EN": [
                "Sweeper burning plastic waste and dry leaves in open plot at {locality} causing toxic smoke.",
                "Open garbage burning causing heavy smog and breathing issues in {locality}.",
                "Industrial smoke emissions from unapproved chimney in {locality} area.",
                "Toxic smoke from burning rubbish creating air pollution hazard in {locality}."
            ],
            "MR": [
                "Kachra aani plastic jalala jaat aahe {locality} madhe, khup dhur hotoe.",
                "Air pollution vadhle aahe open garbage burning mule {locality} madhe.",
                "Plastic burning in open ground causing smoke choking in {locality}.",
                "Dhur aani dhul hazard zali aahe {locality} residentials madhe."
            ],
            "HI": [
                "Kachra aur plastic jalaya ja raha hai {locality} me, dhuan phail raha hai.",
                "Open garbage burning creating severe air pollution in {locality}.",
                "Toxic smoke from trash burning in {locality} open land.",
                "Heavy smog due to burning dry leaves in {locality} area."
            ]
        },
        "PRK-NOIS": {
            "EN": [
                "Heavy industrial generator operating 24/7 causing severe noise pollution in {locality}.",
                "Unbearable commercial machinery noise running at night in {locality} residential zone.",
                "Excessive noise pollution from illegal industrial unit in {locality}.",
                "High noise level from commercial construction generator in quiet zone at {locality}."
            ],
            "MR": [
                "Industrial generator cha motha aavaj hotoe {locality} residential madhe.",
                "Aawajache pradushan khup vadhle aahe commercial machine mule {locality} madhe.",
                "Machine noise at night creating disturbance in {locality}.",
                "Unbearable loud noise from generator in {locality} hospital zone."
            ],
            "HI": [
                "Industrial generator ka shor bahut hai {locality} residential me.",
                "Commercial machine noise causing noise pollution in {locality}.",
                "Noise level very high from heavy machinery in {locality} at night.",
                "Generator noise disturbance in {locality} area."
            ]
        },
        "SAF-FIRE": {
            "EN": [
                "Commercial building emergency staircase and fire exit blocked with goods in {locality}.",
                "Defective fire extinguishers and missing fire safety measures in {locality} complex.",
                "Unsafe storage of highly flammable chemical drums in residential building in {locality}.",
                "Fire safety hazard due to blocked emergency escape route at {locality} mall."
            ],
            "MR": [
                "Fire exit staircase block kela aahe saaman takun {locality} imaratit.",
                "Fire safety norms follow hot nahiye {locality} commercial building madhe.",
                "Flammable chemical unsafe store kela aahe {locality} residential madhe.",
                "Emergency fire exit band aahe {locality} complex jawad."
            ],
            "HI": [
                "Fire exit aur emergency staircase block karke rakha hai {locality} me.",
                "Fire safety equipment non-functional in {locality} commercial complex.",
                "Unsafe storage of chemical barrels in {locality} residential area.",
                "Blocked fire escape hazard in {locality} building."
            ]
        },
        "SAF-DIS": {
            "EN": [
                "Dilapidated building balcony wall cracked severely and leaning over street in {locality}.",
                "Imminent structural collapse risk of old vacant building wall in {locality}.",
                "Crumbling concrete structure hanging dangerously over public sidewalk in {locality}.",
                "Unsafe cracked retaining wall about to collapse on road in {locality}."
            ],
            "MR": [
                "Juny imaratichi bhinta padnyacha dhoka aahe {locality} rastyavar.",
                "Dilapidated structure balcony wall phutli aahe {locality} main street var.",
                "Imminent collapse hazard of old wall in {locality} area.",
                "Bhinta tutun rastyavar padnar aahe {locality} chowk madhe."
            ],
            "HI": [
                "Purani building ki deewar girne ki kagar par hai {locality} main road par.",
                "Dilapidated structure collapse risk in {locality} near market.",
                "Cracked balcony wall hanging dangerously over sidewalk at {locality}.",
                "Unsafe old building wall ready to fall in {locality} area."
            ]
        }
    }

    SPECIAL_STATUS_PATTERNS = {
        "AMBIGUOUS": [
            "There is some big problem near my house in {locality}.",
            "Civic issue on main road near bus stop in {locality}, please check.",
            "Something is wrong near the society gate in {locality}.",
            "Municipal team needed to inspect issue in {locality}.",
            "Rastyavar kahi tari samasya aahe {locality} madhe."
        ],
        "UNKNOWN": [
            "Strange humming sound coming from sky near {locality}.",
            "Unexpected unidentified object lying near open plot in {locality}.",
            "Mysterious purple liquid appearing on street at night in {locality}.",
            "Unexplained phenomenon observed near park in {locality}."
        ],
        "INSUFFICIENT_INFORMATION": [
            "Please solve this issue.",
            "Bad service.",
            "Fix immediately.",
            "Problem near house.",
            "Krupaya bagha.",
            "Solve my complaint.",
            "Help needed."
        ],
        "OUT_OF_SCOPE": [
            "My mobile phone battery is draining very fast.",
            "Cricket match umpire gave incorrect decision yesterday.",
            "Private company electricity bill inside my apartment is high.",
            "Ordered laptop online but delivery is delayed.",
            "My home Wi-Fi internet connection is not working.",
            "Bus ticket price was increased by private operator.",
            "Crude oil pipeline burst near industrial zone in {locality}, oil spilling on road.",
            "LPG natural gas pipeline leaking near petrol pump on highway near {locality}.",
            "AAj aamchya samorcha pipe phutla ani tyatun tel nighayla lagle mag samajla ki to oil pipe hota in {locality}.",
            "Underground chemical factory pipeline burst spilling toxic liquid near {locality}.",
            "Jenkins CI/CD software data deployment pipeline failed during code build.",
            "PyTorch neural network tensor data pipeline crashed due to GPU memory error."
        ]
    }

    def __init__(self, taxonomy_manager: TaxonomyManager = None):
        self.tm = taxonomy_manager or TaxonomyManager()

    def generate_category_dataset(self, target_per_category: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
        """Generates synthetic supervised complaint dataset dynamically reading categories from taxonomy."""
        random.seed(seed)
        all_categories = self.tm.get_categories()
        dataset = []
        doc_id_counter = 1000

        for cat_code, cat_info in all_categories.items():
            # Find domain for this category
            domain_code = None
            for d_code, d_data in self.tm.data.get("domains", {}).items():
                if cat_code in d_data.get("categories", {}):
                    domain_code = d_code
                    break

            dept_map = cat_info.get("department_mapping", {})
            dept_code = dept_map.get("department_code")
            patterns = self.CATEGORY_PATTERNS.get(cat_code, {})

            if not patterns:
                # Fallback template if missing
                patterns = {
                    "EN": [f"{cat_info.get('category_name')} issue reported in {{locality}} {{duration}}."],
                    "MR": [f"{cat_info.get('category_name')} samasya aahe {{locality}} madhe {{duration}}."],
                    "HI": [f"{cat_info.get('category_name')} ka problem hai {{locality}} me {{duration}}."]
                }

            # Generate target_per_category examples
            for i in range(target_per_category):
                # Language distribution: 50% EN, 25% MR, 25% HI
                rand_val = random.random()
                if rand_val < 0.50:
                    lang = "EN"
                    template = random.choice(patterns.get("EN", patterns["EN"]))
                    dur = random.choice(self.DURATIONS_EN)
                    sev = random.choice(self.SEVERITY_EN)
                elif rand_val < 0.75:
                    lang = "MARATHI"
                    template = random.choice(patterns.get("MR", patterns.get("EN", ["Issue in {locality}"])))
                    dur = random.choice(self.DURATIONS_MR)
                    sev = random.choice(self.SEVERITY_MR)
                else:
                    lang = "HINGLISH"
                    template = random.choice(patterns.get("HI", patterns.get("EN", ["Issue in {locality}"])))
                    dur = random.choice(self.DURATIONS_HI)
                    sev = random.choice(self.SEVERITY_HI)

                loc = random.choice(self.PUNE_LOCALITIES)
                text = template.format(locality=loc, duration=dur)

                # Add severity or variation 40% of the time
                if random.random() < 0.40:
                    text = f"{text} {sev}"

                # Add noise/typos 15% of the time
                if random.random() < 0.15:
                    text = text.replace("road", "rod").replace("pothole", "pothol").replace("water", "paani").replace(".", "!!")

                diff = "EASY"
                if i % 5 == 0:
                    diff = "MEDIUM"
                elif i % 10 == 0:
                    diff = "HARD_OVERLAPPING"

                doc_id_counter += 1
                dataset.append({
                    "id": f"CF-SYN-{doc_id_counter}",
                    "complaint_text": text,
                    "domain_code": domain_code,
                    "category_code": cat_code,
                    "department_code": dept_code,
                    "language": lang,
                    "difficulty": diff,
                    "source": "synthetic_generator_v1"
                })

        return dataset

    def generate_special_cases_dataset(self, target_per_status: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
        """Generates special status dataset (AMBIGUOUS, UNKNOWN, INSUFFICIENT_INFORMATION, OUT_OF_SCOPE)."""
        random.seed(seed + 1)
        special_dataset = []
        counter = 5000

        for status, patterns in self.SPECIAL_STATUS_PATTERNS.items():
            for i in range(target_per_status):
                pat = random.choice(patterns)
                loc = random.choice(self.PUNE_LOCALITIES)
                text = pat.format(locality=loc) if "{locality}" in pat else pat

                counter += 1
                special_dataset.append({
                    "id": f"CF-SPEC-{counter}",
                    "complaint_text": text,
                    "domain_code": None,
                    "category_code": None,
                    "department_code": None,
                    "language": "EN" if i % 2 == 0 else "HINGLISH",
                    "difficulty": status,
                    "source": "special_cases_generator"
                })

        return special_dataset


if __name__ == "__main__":
    gen = MunicipalDatasetGenerator()
    cat_ds = gen.generate_category_dataset(target_per_category=100)
    spec_ds = gen.generate_special_cases_dataset(target_per_status=50)

    out_dir = Path(__file__).parent / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "synthetic_complaints.json", "w", encoding="utf-8") as f:
        json.dump(cat_ds, f, indent=2, ensure_ascii=False)

    with open(out_dir / "special_cases.json", "w", encoding="utf-8") as f:
        json.dump(spec_ds, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(cat_ds)} supervised category complaints in raw/synthetic_complaints.json")
    print(f"Generated {len(spec_ds)} special status complaints in raw/special_cases.json")
