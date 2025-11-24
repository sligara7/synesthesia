#!/usr/bin/env python3
"""
Accessibility Applications - Structural Translation for Sensory Domains

Demonstrates how structural synesthesia can help:
1. Blind users: Visual structure → Audio structure
2. Deaf users: Audio structure → Visual structure
3. Anyone: Complex data → Accessible representations

The key: We translate STRUCTURE (relationships, patterns, topology)
not just CONTENT (labels, descriptions).
"""

import sys
sys.path.insert(0, 'src')

import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math


# =============================================================================
# COMMON TYPES
# =============================================================================

@dataclass
class SpatialPosition:
    """Position in 2D/3D space."""
    x: float  # -1 to 1 (left to right)
    y: float  # -1 to 1 (bottom to top)
    z: float = 0  # 0 to 1 (near to far)


@dataclass
class VisualObject:
    """An object in the visual scene."""
    id: str
    position: SpatialPosition
    size: float  # 0 to 1
    category: str  # "person", "furniture", "obstacle", etc.
    properties: Dict = None


@dataclass
class AudioElement:
    """An element in the audio scene."""
    id: str
    pan: float        # -1 (left) to 1 (right)
    frequency: float  # Hz
    volume: float     # 0 to 1
    timbre: str       # "sine", "square", "noise", etc.
    duration: float   # seconds
    properties: Dict = None


# =============================================================================
# VISUAL → AUDIO (For Blind/Low-Vision Users)
# =============================================================================

class VisualToAudioTranslator:
    """
    Translates visual scene structure to audio structure.

    Preserves:
    - Spatial relationships (position → stereo pan + frequency)
    - Size/importance (size → volume + duration)
    - Object types (category → timbre)
    - Connections (proximity → harmonic relationship)
    """

    # Category to timbre mapping
    CATEGORY_TIMBRES = {
        "person": "warm_pad",      # Friendly, human-like
        "furniture": "soft_bell",  # Solid, stable
        "obstacle": "sharp_buzz",  # Warning, attention
        "door": "chime",           # Opportunity, passage
        "wall": "low_drone",       # Boundary, limit
        "path": "gentle_sweep",    # Movement, flow
        "text": "speech_like",     # Information
        "unknown": "neutral_tone",
    }

    # Frequency range for y-position mapping
    FREQ_MIN = 200   # Hz (bottom of scene)
    FREQ_MAX = 2000  # Hz (top of scene)

    def translate_scene(self, visual_objects: List[VisualObject]) -> Tuple[nx.DiGraph, List[AudioElement]]:
        """
        Translate a visual scene to audio representation.

        Returns both:
        - A graph showing structural relationships
        - A list of audio elements with parameters
        """
        # Build visual scene graph
        visual_graph = self._build_visual_graph(visual_objects)

        # Translate to audio
        audio_elements = []
        for obj in visual_objects:
            audio = self._translate_object(obj)
            audio_elements.append(audio)

        # Build audio graph (preserving structure)
        audio_graph = self._build_audio_graph(visual_graph, audio_elements)

        return audio_graph, audio_elements

    def _build_visual_graph(self, objects: List[VisualObject]) -> nx.DiGraph:
        """Build graph of spatial relationships between objects."""
        G = nx.DiGraph()

        # Add nodes
        for obj in objects:
            G.add_node(obj.id,
                      category=obj.category,
                      position=(obj.position.x, obj.position.y, obj.position.z),
                      size=obj.size)

        # Add edges based on proximity
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                distance = self._spatial_distance(obj1.position, obj2.position)
                if distance < 0.5:  # Close objects are connected
                    G.add_edge(obj1.id, obj2.id,
                              distance=distance,
                              relationship="near")
                    G.add_edge(obj2.id, obj1.id,
                              distance=distance,
                              relationship="near")

        return G

    def _translate_object(self, obj: VisualObject) -> AudioElement:
        """Translate a single visual object to audio element."""

        # Position → Pan (left-right)
        pan = obj.position.x

        # Position → Frequency (up-down)
        # Higher objects = higher frequency
        freq_range = self.FREQ_MAX - self.FREQ_MIN
        frequency = self.FREQ_MIN + (obj.position.y + 1) / 2 * freq_range

        # Depth → Volume (closer = louder)
        volume = 1.0 - obj.position.z * 0.5

        # Size → Duration (bigger = longer)
        duration = 0.1 + obj.size * 0.4

        # Category → Timbre
        timbre = self.CATEGORY_TIMBRES.get(obj.category, "neutral_tone")

        return AudioElement(
            id=f"audio_{obj.id}",
            pan=pan,
            frequency=frequency,
            volume=volume,
            timbre=timbre,
            duration=duration,
            properties={
                "original_category": obj.category,
                "original_position": (obj.position.x, obj.position.y, obj.position.z)
            }
        )

    def _build_audio_graph(self, visual_graph: nx.DiGraph,
                           audio_elements: List[AudioElement]) -> nx.DiGraph:
        """Build audio graph preserving visual structure."""
        G = nx.DiGraph()

        # Map visual IDs to audio IDs
        id_map = {ae.properties["original_category"]: ae.id for ae in audio_elements}

        # Add nodes
        for ae in audio_elements:
            G.add_node(ae.id,
                      pan=ae.pan,
                      frequency=ae.frequency,
                      volume=ae.volume,
                      timbre=ae.timbre)

        # Copy edges (structural relationships)
        for u, v, data in visual_graph.edges(data=True):
            audio_u = f"audio_{u}"
            audio_v = f"audio_{v}"
            if audio_u in G.nodes and audio_v in G.nodes:
                # Translate spatial distance to harmonic relationship
                distance = data.get("distance", 0.5)
                harmonic_interval = self._distance_to_interval(distance)
                G.add_edge(audio_u, audio_v,
                          harmonic_interval=harmonic_interval,
                          relationship="harmonic_" + data.get("relationship", "related"))

        return G

    def _spatial_distance(self, p1: SpatialPosition, p2: SpatialPosition) -> float:
        """Euclidean distance between two positions."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _distance_to_interval(self, distance: float) -> str:
        """Map spatial distance to musical interval."""
        # Closer objects = more consonant intervals
        if distance < 0.2:
            return "unison"
        elif distance < 0.3:
            return "third"
        elif distance < 0.4:
            return "fifth"
        else:
            return "seventh"


# =============================================================================
# AUDIO → VISUAL (For Deaf/Hard-of-Hearing Users)
# =============================================================================

class AudioToVisualTranslator:
    """
    Translates audio structure to visual representation.

    Preserves:
    - Sound positions → Visual positions
    - Volume dynamics → Size/brightness
    - Frequency → Color/height
    - Rhythm → Animation timing
    - Conversation flow → Arrow/flow diagrams
    """

    def translate_conversation(self, speakers: List[Dict]) -> nx.DiGraph:
        """
        Translate conversation structure to visual graph.

        speakers: List of {id, turns: [{start, end, emotion, volume}]}
        """
        G = nx.DiGraph()

        # Add speaker nodes
        for speaker in speakers:
            G.add_node(speaker["id"],
                      type="speaker",
                      color=self._speaker_color(speaker["id"]))

        # Add turn nodes and edges
        all_turns = []
        for speaker in speakers:
            for i, turn in enumerate(speaker.get("turns", [])):
                turn_id = f"{speaker['id']}_turn_{i}"
                G.add_node(turn_id,
                          type="speech_turn",
                          start=turn.get("start", 0),
                          end=turn.get("end", 1),
                          emotion=turn.get("emotion", "neutral"),
                          volume=turn.get("volume", 0.5),
                          speaker=speaker["id"])

                # Connect to speaker
                G.add_edge(speaker["id"], turn_id, relation="speaks")
                all_turns.append((turn.get("start", 0), turn_id, speaker["id"]))

        # Add sequential edges (conversation flow)
        all_turns.sort(key=lambda x: x[0])
        for i in range(len(all_turns) - 1):
            current = all_turns[i]
            next_turn = all_turns[i + 1]

            # Determine flow type
            if current[2] == next_turn[2]:
                flow_type = "continues"
            else:
                flow_type = "responds"

            G.add_edge(current[1], next_turn[1],
                      relation=flow_type,
                      gap=next_turn[0] - current[0])

        return G

    def translate_soundscape(self, sounds: List[Dict]) -> nx.DiGraph:
        """
        Translate environmental sounds to visual layout.

        sounds: List of {id, position, frequency, volume, type}
        """
        G = nx.DiGraph()

        for sound in sounds:
            # Frequency → vertical position
            y = (sound.get("frequency", 500) - 200) / 1800  # Normalize to 0-1

            # Stereo position → horizontal position
            x = sound.get("position", 0)

            # Volume → size
            size = sound.get("volume", 0.5)

            # Type → shape/color
            visual_type = self._sound_to_visual_type(sound.get("type", "unknown"))

            G.add_node(sound["id"],
                      x=x, y=y,
                      size=size,
                      visual_type=visual_type,
                      original_sound=sound.get("type", "unknown"))

        # Add proximity relationships
        sound_list = list(sounds)
        for i, s1 in enumerate(sound_list):
            for s2 in sound_list[i+1:]:
                # If sounds are related (e.g., both from same source)
                if abs(s1.get("position", 0) - s2.get("position", 0)) < 0.3:
                    G.add_edge(s1["id"], s2["id"], relation="co-located")

        return G

    def _speaker_color(self, speaker_id: str) -> str:
        """Assign consistent color to speaker."""
        colors = ["blue", "green", "orange", "purple", "red", "teal"]
        return colors[hash(speaker_id) % len(colors)]

    def _sound_to_visual_type(self, sound_type: str) -> str:
        """Map sound type to visual representation."""
        mapping = {
            "speech": "speech_bubble",
            "music": "wave_pattern",
            "alarm": "exclamation",
            "notification": "badge",
            "ambient": "gradient_cloud",
            "footstep": "ripple",
            "door": "rectangle_pulse",
        }
        return mapping.get(sound_type, "circle")


# =============================================================================
# DEMO
# =============================================================================

def demo_visual_to_audio():
    """Demonstrate visual→audio translation for blind users."""
    print("\n" + "=" * 70)
    print("VISUAL → AUDIO TRANSLATION (For Blind/Low-Vision Users)")
    print("=" * 70)

    print("""
Scenario: A blind person enters a room. Instead of describing
"there's a chair, then a table, then a door," we translate the
STRUCTURE of the room into audio that preserves spatial relationships.
    """)

    # Define a room scene
    objects = [
        VisualObject("chair", SpatialPosition(-0.6, -0.3, 0.2), 0.3, "furniture"),
        VisualObject("table", SpatialPosition(0.0, 0.0, 0.3), 0.5, "furniture"),
        VisualObject("person", SpatialPosition(0.4, 0.2, 0.4), 0.4, "person"),
        VisualObject("door", SpatialPosition(0.8, 0.0, 0.6), 0.4, "door"),
        VisualObject("obstacle", SpatialPosition(-0.2, -0.5, 0.1), 0.2, "obstacle"),
    ]

    print("Visual Scene (what a sighted person sees):")
    print("─" * 50)
    print("""
    ┌─────────────────────────────────────┐
    │                        👤 person    │  (right, slightly up)
    │                                     │
    │     🪑 chair    🪑 table      🚪 door│  (left, center, right)
    │                                     │
    │         ⚠️ obstacle                 │  (center-left, down)
    └─────────────────────────────────────┘
    """)

    # Translate
    translator = VisualToAudioTranslator()
    audio_graph, audio_elements = translator.translate_scene(objects)

    print("\nAudio Translation (what the blind person hears):")
    print("─" * 50)

    for ae in audio_elements:
        orig = ae.properties.get("original_category", "unknown")
        pan_desc = "left" if ae.pan < -0.3 else "right" if ae.pan > 0.3 else "center"
        freq_desc = "low" if ae.frequency < 600 else "high" if ae.frequency > 1200 else "mid"

        print(f"\n  {orig.upper()}:")
        print(f"    Position: {pan_desc} ({ae.pan:.1f})")
        print(f"    Height: {freq_desc} ({ae.frequency:.0f} Hz)")
        print(f"    Proximity: {'close' if ae.volume > 0.7 else 'far'} (vol: {ae.volume:.2f})")
        print(f"    Sound type: {ae.timbre}")

    print("\n\nStructural Relationships Preserved:")
    print("─" * 50)
    for u, v, data in audio_graph.edges(data=True):
        print(f"  {u} ←→ {v}: {data.get('harmonic_interval', 'related')}")

    print("""
KEY INSIGHT:
  The blind user doesn't just hear "chair, table, person, door"
  They hear a SOUNDSCAPE where:
  - Spatial positions are preserved (left/right pan, up/down frequency)
  - Close objects have consonant harmonic relationships
  - Object types have distinct timbres
  - The STRUCTURE of the room is audible, not just a list of objects
    """)


def demo_audio_to_visual():
    """Demonstrate audio→visual translation for deaf users."""
    print("\n" + "=" * 70)
    print("AUDIO → VISUAL TRANSLATION (For Deaf/Hard-of-Hearing Users)")
    print("=" * 70)

    print("""
Scenario: A deaf person is in a meeting. Instead of just showing
captions, we visualize the STRUCTURE of the conversation - who's
talking, the flow of dialogue, emotional dynamics.
    """)

    # Define a conversation
    speakers = [
        {
            "id": "Alice",
            "turns": [
                {"start": 0, "end": 2, "emotion": "curious", "volume": 0.6},
                {"start": 5, "end": 7, "emotion": "agreeing", "volume": 0.5},
                {"start": 12, "end": 15, "emotion": "excited", "volume": 0.8},
            ]
        },
        {
            "id": "Bob",
            "turns": [
                {"start": 2.5, "end": 4.5, "emotion": "explaining", "volume": 0.7},
                {"start": 7.5, "end": 11, "emotion": "elaborating", "volume": 0.6},
            ]
        }
    ]

    print("Audio Structure (what a hearing person experiences):")
    print("─" * 50)
    print("""
    Time: 0──────5──────10─────15

    Alice: ▓▓▓░░░▓▓▓░░░░░░░░▓▓▓▓▓  (curious→agreeing→excited)
    Bob:   ░░░▓▓▓▓░░░▓▓▓▓▓▓░░░░░░  (explaining→elaborating)

    Flow: Alice asks → Bob explains → Alice agrees → Bob elaborates → Alice excited
    """)

    translator = AudioToVisualTranslator()
    conversation_graph = translator.translate_conversation(speakers)

    print("\nVisual Translation (what the deaf person sees):")
    print("─" * 50)
    print("""
    ┌─────────────────────────────────────────────────────┐
    │                                                     │
    │  ┌───────┐          ┌───────┐          ┌───────┐   │
    │  │ Alice │──asks──▶│ Bob   │──respond─▶│ Alice │   │
    │  │curious│          │explain│          │agree  │   │
    │  └───────┘          └───┬───┘          └───────┘   │
    │                         │                    ▲      │
    │                         │                    │      │
    │                    elaborate                 │      │
    │                         │                    │      │
    │                         ▼                    │      │
    │                    ┌───────┐           ┌────┴──┐   │
    │                    │ Bob   │──triggers─▶│Alice │   │
    │                    │detail │           │excited│   │
    │                    └───────┘           └───────┘   │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    Visual indicators:
    • Box size = speaking volume
    • Color intensity = emotional intensity
    • Arrow thickness = conversation flow strength
    • Position = temporal sequence
    """)

    print("\nGraph Structure (preserved from audio):")
    print("─" * 50)
    for node, data in conversation_graph.nodes(data=True):
        if data.get("type") == "speech_turn":
            print(f"  {node}: {data.get('emotion', 'neutral')} (vol: {data.get('volume', 0):.1f})")

    print("\nConversation Flow:")
    for u, v, data in conversation_graph.edges(data=True):
        if data.get("relation") in ["continues", "responds"]:
            print(f"  {u} ──{data['relation']}──▶ {v}")

    print("""
KEY INSIGHT:
  The deaf user doesn't just see text captions.
  They see the STRUCTURE of the conversation:
  - Who's responding to whom (flow arrows)
  - Emotional dynamics (color/size)
  - Turn-taking rhythm (spatial layout)
  - The PATTERN of dialogue, not just the words
    """)


def demo_data_accessibility():
    """Demonstrate complex data → accessible format."""
    print("\n" + "=" * 70)
    print("COMPLEX DATA → ACCESSIBLE REPRESENTATION")
    print("=" * 70)

    print("""
Scenario: Making charts and data visualizations accessible to
blind users through structural audio representation.
    """)

    # A simple bar chart as a graph
    chart_data = nx.DiGraph()

    # Add data points as nodes
    values = [("Q1", 45), ("Q2", 72), ("Q3", 58), ("Q4", 91)]

    print("Original Chart (visual):")
    print("─" * 50)
    print("""
    Sales by Quarter

    Q1  ████████████████░░░░░░░░░░░░░░░░  45
    Q2  ██████████████████████████░░░░░░  72
    Q3  █████████████████████░░░░░░░░░░░  58
    Q4  █████████████████████████████████  91
        0    20    40    60    80   100
    """)

    for label, value in values:
        chart_data.add_node(label, value=value, normalized=value/100)

    # Add sequential relationships
    for i in range(len(values) - 1):
        current = values[i]
        next_val = values[i + 1]
        change = next_val[1] - current[1]
        chart_data.add_edge(current[0], next_val[0],
                           change=change,
                           direction="up" if change > 0 else "down")

    print("\nAudio Translation (structural):")
    print("─" * 50)
    print("""
    Instead of: "Q1 is 45, Q2 is 72, Q3 is 58, Q4 is 91"

    Audio structure:

    Q1: ♩ (low tone, short duration)           [45 = low]
         ↗ ascending glide
    Q2: ♩♩ (higher tone, longer duration)      [72 = grew significantly]
         ↘ descending glide
    Q3: ♩ (mid tone, medium duration)          [58 = dropped]
         ↗ strong ascending glide
    Q4: ♩♩♩ (high tone, longest duration)      [91 = peak!]

    The PATTERN is audible:
    - Rise → Rise → Dip → Strong Rise
    - Not just values, but RELATIONSHIPS between values
    """)

    print("\nStructural Relationships:")
    print("─" * 50)
    for u, v, data in chart_data.edges(data=True):
        print(f"  {u} → {v}: {data['direction']} by {abs(data['change'])}")

    print("""
KEY INSIGHT:
  A blind user hearing this audio representation understands:
  - The overall TREND (rising with a dip)
  - The relative MAGNITUDES (Q4 is biggest)
  - The CHANGES between periods (Q2 was big growth)

  This is richer than a list of numbers because
  it preserves the STRUCTURE of the data.
    """)


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   STRUCTURAL SYNESTHESIA FOR ACCESSIBILITY                               ║
║                                                                          ║
║   Translating STRUCTURE across sensory domains to help everyone          ║
║   experience information in the way that works best for them.            ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
    """)

    demo_visual_to_audio()
    demo_audio_to_visual()
    demo_data_accessibility()

    print("\n" + "=" * 70)
    print("THE VISION")
    print("=" * 70)
    print("""
Structural Synesthesia for accessibility is about more than translation.
It's about PRESERVING EXPERIENCE across sensory domains.

Traditional:  "There's a sunset with orange and red colors"
Structural:   *Audio that conveys the radiating structure, the gradient,
               the horizon line, the relationship between elements*

Traditional:  "Two people are having an argument"
Structural:   *Visual flow that shows the back-and-forth rhythm,
               the escalating intensity, the interruption patterns*

The goal: Not to describe what others experience,
          but to let everyone EXPERIENCE the same structures
          through whatever sensory channels work for them.

This is why STRUCTURE matters more than CONTENT.
Structure can be translated. Experience can be shared.
    """)


if __name__ == "__main__":
    main()
