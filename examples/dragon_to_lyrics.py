#!/usr/bin/env python3
"""
Dragon to Lyrics - Creative Cross-Domain Translation

This demo takes the visual structure of a dragon image and translates
it into musical structure, then generates lyrics that follow that pattern.

The magic: We're not translating MEANING, we're translating STRUCTURE.
The dragon's visual composition becomes the song's architecture.
"""

import sys
sys.path.insert(0, 'src')

import networkx as nx
from structural_rorschach import create_service_container


def create_dragon_image_graph():
    """
    Create a graph representing the visual structure of a dragon image.

    Imagine a dramatic dragon image:
    - Central body (dominant focal point)
    - Wings spread wide (radiating elements)
    - Tail curving (flowing line)
    - Fire breath (dynamic outward burst)
    - Scales/texture (repeating pattern)
    - Eyes (secondary focal points)
    """
    G = nx.DiGraph()

    # === COMPOSITION HIERARCHY ===
    # The dragon's body is the central hub
    G.add_node("body",
               role="focal_point",
               visual_weight=10,
               description="Central mass, dominant element")

    # Wings radiate from body (power, expansion)
    G.add_node("left_wing", role="radiating", visual_weight=7)
    G.add_node("right_wing", role="radiating", visual_weight=7)
    G.add_edge("body", "left_wing", relation="extends")
    G.add_edge("body", "right_wing", relation="extends")

    # Wing details (feathered structure)
    for wing in ["left_wing", "right_wing"]:
        for i in range(3):
            feather = f"{wing}_segment_{i}"
            G.add_node(feather, role="detail", visual_weight=2)
            G.add_edge(wing, feather, relation="contains")

    # Head connects to body, contains eyes and fire
    G.add_node("head", role="secondary_focal", visual_weight=8)
    G.add_edge("body", "head", relation="connects")

    # Eyes - intense focal points
    G.add_node("left_eye", role="intensity_point", visual_weight=5)
    G.add_node("right_eye", role="intensity_point", visual_weight=5)
    G.add_edge("head", "left_eye", relation="contains")
    G.add_edge("head", "right_eye", relation="contains")

    # Fire breath - dynamic outward flow
    G.add_node("fire_origin", role="emission_point", visual_weight=6)
    G.add_edge("head", "fire_origin", relation="emits")

    # Fire expands outward (crescendo pattern)
    fire_stages = ["fire_spark", "fire_flame", "fire_blaze", "fire_inferno"]
    prev = "fire_origin"
    for stage in fire_stages:
        G.add_node(stage, role="expanding", visual_weight=4)
        G.add_edge(prev, stage, relation="grows_into")
        prev = stage

    # Tail - flowing curved line (melodic)
    G.add_node("tail_base", role="flowing", visual_weight=4)
    G.add_edge("body", "tail_base", relation="extends")

    tail_segments = ["tail_mid", "tail_curve", "tail_tip"]
    prev = "tail_base"
    for seg in tail_segments:
        G.add_node(seg, role="flowing", visual_weight=3)
        G.add_edge(prev, seg, relation="flows_to")
        prev = seg

    # Claws - grounding elements
    for i in range(4):
        claw = f"claw_{i}"
        G.add_node(claw, role="grounding", visual_weight=2)
        G.add_edge("body", claw, relation="grounds")

    return G


def create_musical_structure_corpus():
    """Create a corpus of musical/lyrical structures."""

    structures = {}

    # === VERSE-CHORUS-BRIDGE (Pop/Rock) ===
    vcb = nx.DiGraph()
    vcb.add_node("song", role="container")
    for section in ["verse1", "chorus1", "verse2", "chorus2", "bridge", "chorus3"]:
        vcb.add_node(section, role="section")
        vcb.add_edge("song", section)
    # Linear flow
    sections = ["verse1", "chorus1", "verse2", "chorus2", "bridge", "chorus3"]
    for i in range(len(sections)-1):
        vcb.add_edge(sections[i], sections[i+1], relation="leads_to")
    structures["Verse-Chorus-Bridge"] = vcb

    # === BUILDING ANTHEM (Crescendo) ===
    anthem = nx.DiGraph()
    anthem.add_node("quiet_intro", role="beginning", intensity=1)
    anthem.add_node("building_verse", role="development", intensity=3)
    anthem.add_node("pre_chorus", role="tension", intensity=5)
    anthem.add_node("explosive_chorus", role="climax", intensity=10)
    anthem.add_node("soaring_bridge", role="peak", intensity=9)
    anthem.add_node("triumphant_outro", role="resolution", intensity=8)

    flow = ["quiet_intro", "building_verse", "pre_chorus",
            "explosive_chorus", "soaring_bridge", "triumphant_outro"]
    for i in range(len(flow)-1):
        anthem.add_edge(flow[i], flow[i+1], relation="builds_to")

    # Chorus has radiating hooks
    for hook in ["hook_melody", "hook_rhythm", "hook_lyric"]:
        anthem.add_node(hook, role="hook")
        anthem.add_edge("explosive_chorus", hook)
    structures["Building Anthem"] = anthem

    # === POWER BALLAD ===
    ballad = nx.DiGraph()
    ballad.add_node("intimate_intro", role="quiet", intensity=2)
    ballad.add_node("storytelling_verse1", role="narrative", intensity=3)
    ballad.add_node("emotional_prechorus", role="building", intensity=5)
    ballad.add_node("powerful_chorus", role="release", intensity=8)
    ballad.add_node("deeper_verse2", role="narrative", intensity=4)
    ballad.add_node("soaring_bridge", role="climax", intensity=10)
    ballad.add_node("gentle_outro", role="resolution", intensity=3)

    flow = ["intimate_intro", "storytelling_verse1", "emotional_prechorus",
            "powerful_chorus", "deeper_verse2", "soaring_bridge", "gentle_outro"]
    for i in range(len(flow)-1):
        ballad.add_edge(flow[i], flow[i+1])
    structures["Power Ballad"] = ballad

    # === EPIC METAL ===
    metal = nx.DiGraph()
    metal.add_node("dramatic_intro", role="establishing", intensity=7)
    metal.add_node("crushing_riff", role="power", intensity=9)
    metal.add_node("verse_assault", role="verse", intensity=8)
    metal.add_node("melodic_break", role="contrast", intensity=4)
    metal.add_node("chorus_anthem", role="chorus", intensity=10)
    metal.add_node("solo_section", role="virtuosity", intensity=9)
    metal.add_node("breakdown", role="heaviest", intensity=10)
    metal.add_node("triumphant_end", role="finale", intensity=10)

    # Hub structure - riff connects to everything
    metal.add_edge("dramatic_intro", "crushing_riff")
    metal.add_edge("crushing_riff", "verse_assault")
    metal.add_edge("crushing_riff", "chorus_anthem")
    metal.add_edge("verse_assault", "melodic_break")
    metal.add_edge("melodic_break", "chorus_anthem")
    metal.add_edge("chorus_anthem", "solo_section")
    metal.add_edge("solo_section", "breakdown")
    metal.add_edge("breakdown", "triumphant_end")

    # Breakdown has multiple elements
    for element in ["double_bass", "palm_mute", "scream"]:
        metal.add_node(element, role="texture")
        metal.add_edge("breakdown", element)
    structures["Epic Metal"] = metal

    # === PROGRESSIVE EPIC ===
    prog = nx.DiGraph()
    prog.add_node("overture", role="introduction", intensity=5)

    # Multiple movements
    movements = ["movement_i", "movement_ii", "movement_iii", "finale"]
    prog.add_edge("overture", movements[0])
    for i in range(len(movements)-1):
        prog.add_node(movements[i], role="movement")
        prog.add_edge(movements[i], movements[i+1])
    prog.add_node("finale", role="climax")

    # Each movement has themes
    for mov in movements[:3]:
        for theme in ["theme_a", "theme_b"]:
            node = f"{mov}_{theme}"
            prog.add_node(node, role="theme")
            prog.add_edge(mov, node)

    # Finale brings back all themes
    for mov in movements[:3]:
        prog.add_edge(f"{mov}_theme_a", "finale", relation="returns")

    structures["Progressive Epic"] = prog

    return structures


def analyze_and_match(dragon_graph, music_structures):
    """Find which musical structure best matches the dragon's visual structure."""

    container = create_service_container()
    extractor = container.signature_extractor
    corpus_service = container.corpus_service
    resonance_service = container.resonance_service

    # Build music corpus
    corpus = corpus_service.create_corpus("music_structures", "music")
    music_sigs = {}

    for name, graph in music_structures.items():
        sig = extractor.extract_from_networkx(graph, "music", name, name.lower().replace(" ", "_"))
        corpus_service.add_signature(corpus, sig)
        music_sigs[name] = sig

    # Extract dragon signature
    dragon_sig = extractor.extract_from_networkx(
        dragon_graph, "image", "Dragon Image", "dragon"
    )

    # Find resonances
    resonances = resonance_service.find_resonances(
        dragon_sig, corpus, top_k=5, threshold=0.0
    )

    return dragon_sig, resonances, music_sigs


def generate_lyrics_structure(matched_form: str, dragon_elements: dict):
    """
    Generate a lyrical structure that matches both:
    - The musical form we matched to
    - The dragon's visual elements translated to themes

    This maps visual elements to lyrical themes:
    - Body (central) → Core message/identity
    - Wings (radiating) → Freedom, power, expansion
    - Fire (crescendo) → Passion, destruction, transformation
    - Tail (flowing) → Journey, movement, grace
    - Eyes (focal) → Vision, intensity, soul
    - Claws (grounding) → Strength, earth, foundation
    """

    print("\n" + "="*60)
    print("GENERATING LYRICS STRUCTURE")
    print("="*60)

    # Theme mapping from visual to lyrical
    themes = {
        "body": "identity, presence, being",
        "wings": "freedom, soaring, power",
        "fire": "passion, transformation, fury",
        "tail": "journey, grace, flow",
        "eyes": "vision, intensity, soul",
        "claws": "strength, grounding, earth"
    }

    print(f"\nMatched Musical Form: {matched_form}")
    print("\nVisual → Lyrical Theme Mapping:")
    for visual, lyrical in themes.items():
        print(f"  {visual:8} → {lyrical}")

    if "Metal" in matched_form or "Epic" in matched_form:
        print("\n" + "-"*50)
        print("SONG STRUCTURE: Epic Metal Dragon")
        print("-"*50)
        print("""
[DRAMATIC INTRO - Orchestral/Atmospheric]
  Visual: Dragon silhouette emerging from darkness
  Theme: Anticipation, mystery

[VERSE 1 - Building intensity]
  Visual: Body and wings revealed
  Theme: Identity and power awakening

  "From ancient depths where shadows sleep
   A force of nature starts to rise
   Scales like armor, dark and deep
   Fire burning in these eyes"

[PRE-CHORUS - Tension building]
  Visual: Eyes lock on viewer, fire gathering
  Theme: Intensity, imminent transformation

  "Feel the heat begin to grow
   Wings unfold against the sky"

[CHORUS - Full power, radiating]
  Visual: Wings spread wide, fire erupting
  Theme: Freedom, power unleashed

  "I am the fire, I am the flame
   Born of fury, wild, untamed
   Soaring higher, burning bright
   Dragon heart ignites the night"

[VERSE 2 - Movement and grace]
  Visual: Tail sweeping, body in motion
  Theme: Journey, unstoppable force

  "Through the storm I carve my path
   Tail like lightning, swift and true
   Mountains crumble in my wrath
   Ancient skies turn crimson hue"

[BRIDGE - Peak intensity]
  Visual: Full dragon glory, maximum fire
  Theme: Transformation complete

  "No chain can hold, no cage contain
   This heart that beats with primal flame"

[FINAL CHORUS - Triumphant]
  Visual: Dragon ascending, victorious
  Theme: Transcendence

[OUTRO - Resolution]
  Visual: Dragon disappearing into clouds/stars
  Theme: Legend, eternal
""")

    elif "Ballad" in matched_form:
        print("\n" + "-"*50)
        print("SONG STRUCTURE: Power Ballad Dragon")
        print("-"*50)
        print("""
[INTIMATE INTRO - Piano/Acoustic]
  Visual: Single eye in darkness
  Theme: Mystery, vulnerability

[VERSE 1 - Storytelling]
  Visual: Dragon form slowly revealed
  Theme: Origin, loneliness

  "In caverns deep I wait alone
   Guarding treasures, heart of stone
   Wings folded tight against the cold
   A story that was never told"

[PRE-CHORUS - Emotion building]
  Visual: Wings beginning to stir
  Theme: Longing for freedom

  "But somewhere in this ancient chest
   A spark remains that will not rest"

[CHORUS - Emotional release]
  Visual: Wings spread, gentle fire glow
  Theme: Hope, inner strength

  "I will rise on wings of light
   Break these chains, reclaim the night
   Fire within my soul still burns
   The dragon heart forever yearns"

[VERSE 2 - Deeper narrative]
  Visual: Tail curving gracefully, movement
  Theme: Journey inward

[SOARING BRIDGE - Climax]
  Visual: Full majestic form
  Theme: Self-acceptance, power

[GENTLE OUTRO]
  Visual: Dragon at peace
  Theme: Resolution, wisdom
""")

    else:
        print("\n" + "-"*50)
        print("SONG STRUCTURE: Progressive Epic Dragon")
        print("-"*50)
        print("""
[MOVEMENT I: AWAKENING]
  Visual: Eyes opening, first breath
  Theme A: Ancient power stirring
  Theme B: World sensing the presence

[MOVEMENT II: ASCENSION]
  Visual: Wings unfurling, body rising
  Theme A: Breaking free of earth
  Theme B: Claiming the sky

[MOVEMENT III: INFERNO]
  Visual: Fire breath at full force
  Theme A: Transformation through flame
  Theme B: Destruction and creation

[FINALE: ETERNAL]
  Visual: Dragon among the stars
  All themes return and resolve
  Theme: Legend becomes myth becomes truth
""")

    return themes


def print_structural_analysis(dragon_graph):
    """Print the dragon's structural analysis."""

    print("\n" + "="*60)
    print("DRAGON IMAGE STRUCTURAL ANALYSIS")
    print("="*60)

    # Count structural elements
    roles = {}
    for node, data in dragon_graph.nodes(data=True):
        role = data.get('role', 'unknown')
        roles[role] = roles.get(role, 0) + 1

    print(f"\nNodes: {dragon_graph.number_of_nodes()}")
    print(f"Edges: {dragon_graph.number_of_edges()}")
    print(f"Density: {nx.density(dragon_graph):.3f}")

    print("\nVisual Role Distribution:")
    for role, count in sorted(roles.items(), key=lambda x: -x[1]):
        print(f"  {role:20} : {count}")

    # Find hubs (compositional focal points)
    out_degrees = dict(dragon_graph.out_degree())
    hubs = sorted(out_degrees.items(), key=lambda x: -x[1])[:5]
    print("\nCompositional Hubs (visual focal points):")
    for node, degree in hubs:
        print(f"  {node:20} : {degree} connections")

    # Find flow patterns
    chains = []
    for node in dragon_graph.nodes():
        if dragon_graph.in_degree(node) == 1 and dragon_graph.out_degree(node) == 1:
            chains.append(node)
    print(f"\nFlow Elements (visual movement): {len(chains)} nodes")

    # Radiating patterns
    radiating = [n for n, d in out_degrees.items() if d >= 3]
    print(f"Radiating Elements (expansion): {len(radiating)} nodes")


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   🐉  DRAGON TO LYRICS  🎵                                               ║
║                                                                          ║
║   Translating Visual Structure to Musical/Lyrical Form                   ║
║                                                                          ║
║   "Structure transcends medium - a dragon's form can become a song"      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

    # Create the dragon image structure
    print("Creating dragon image graph structure...")
    dragon_graph = create_dragon_image_graph()

    # Analyze the dragon's visual structure
    print_structural_analysis(dragon_graph)

    # Create musical structure corpus
    print("\n" + "="*60)
    print("MUSICAL STRUCTURE CORPUS")
    print("="*60)
    music_structures = create_musical_structure_corpus()

    print(f"\nAvailable musical forms:")
    for name, graph in music_structures.items():
        print(f"  {name}: {graph.number_of_nodes()} elements, {graph.number_of_edges()} connections")

    # Find the best match
    print("\n" + "="*60)
    print("FINDING STRUCTURAL RESONANCE")
    print("="*60)

    dragon_sig, resonances, music_sigs = analyze_and_match(dragon_graph, music_structures)

    print("\nDragon image structural signature:")
    print(f"  Motif pattern: {list(dragon_sig.motif_vector.keys())[:5]}")
    print(f"  Hub ratio: {dragon_sig.hub_ratio:.2%}")
    print(f"  Is DAG: {dragon_sig.is_dag}")

    print("\n" + "-"*50)
    print("CROSS-DOMAIN MATCHES (Image → Music)")
    print("-"*50)

    for i, res in enumerate(resonances, 1):
        print(f"\n  {i}. {res.match_name}")
        print(f"     Structural similarity: {res.overall_score:.1%}")
        print(f"     Motif match: {res.motif_similarity:.1%}")
        if res.matching_motifs:
            print(f"     Shared patterns: {', '.join(res.matching_motifs[:3])}")

    # Get the best match
    best_match = resonances[0].match_name
    print(f"\n  ★ BEST MATCH: {best_match} ({resonances[0].overall_score:.1%})")

    # Generate lyrics based on the match
    dragon_elements = {
        "body": "central presence",
        "wings": "radiating power",
        "fire": "crescendo intensity",
        "tail": "flowing movement",
        "eyes": "focal intensity",
        "claws": "grounding force"
    }

    generate_lyrics_structure(best_match, dragon_elements)

    print("\n" + "="*60)
    print("WHAT JUST HAPPENED")
    print("="*60)
    print("""
We translated a DRAGON IMAGE into SONG LYRICS through STRUCTURE:

1. VISUAL ANALYSIS
   - Mapped the dragon's composition to a graph
   - Body = hub, Wings = radiating, Fire = crescendo, Tail = flow

2. STRUCTURAL FINGERPRINT
   - Extracted topology: hubs, chains, density, motifs
   - This fingerprint is domain-agnostic

3. CROSS-DOMAIN MATCHING
   - Compared dragon's structure to musical forms
   - Found the song structure with matching topology

4. CREATIVE TRANSLATION
   - Visual elements → Lyrical themes
   - Musical structure → Song architecture

The dragon's visual COMPOSITION became the song's STRUCTURE.
Not through meaning, but through pure topology!
""")


if __name__ == "__main__":
    main()
