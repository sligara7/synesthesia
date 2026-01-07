"""
Structural Rorschach - Cross-Domain Graph Analysis

A system for finding structural resonances across domains (images, music, text)
by comparing graph topologies rather than semantic content.

Inspired by:
- Synesthesia: Cross-sensory perception
- Rorschach inkblot test: Structural interpretation

Core concept: Any structured data can be the "inkblot" that the system
interprets through other domain corpora by matching graph patterns.

Key components:
- StructuralSignature: Full graph analysis (motif-based)
- SpectralSignature: Compressed analysis using linear algebra (scales to any size)
- GraphPruner: Remove noise to focus on essential structure
- SimilarityService: Compute structural similarity
- CorpusService: Manage signature collections
- ResonanceService: Find cross-domain matches
- InterpretationService: Generate human-readable explanations
- DomainAdapters: Convert domain data to graphs
"""

# Core signature types
from .signature import StructuralSignature, Resonance

# Extractors
from .extractor import SignatureExtractor
from .spectral import SpectralSignature, SpectralExtractor, spectral_similarity

# Motif detection
from .motifs import MotifDetector

# Pruning
from .pruning import GraphPruner, auto_prune, PruningStats

# Similarity computation
from .similarity import (
    SimilarityService,
    SimilarityResult,
    compute_similarity
)

# Corpus management
from .corpus import (
    Corpus,
    CorpusService,
    create_corpus,
    load_corpus,
    save_corpus
)

# Resonance finding
from .resonance import (
    ResonanceService,
    find_resonances,
    find_cross_domain_resonances
)

# Interpretation
from .interpretation import (
    InterpretationService,
    explain_resonance,
    generate_comparison_report
)

# Domain adapters
from .adapters import (
    DomainAdapter,
    TextAdapter,
    ImageAdapter,
    MusicAdapter,
    CodeAdapter,
    get_adapter,
    adapt_to_graph,
    validate_graph,
    GraphValidationResult
)

# Protocol definitions (interfaces for dependency injection)
from .protocols import (
    # Data types
    SignatureData,
    ResonanceData,
    SimilarityData,
    # Domain adapter protocols
    CanAdaptToGraph,
    ProvidesGraphValidation,
    # Signature extraction protocols
    CanExtractSignatures,
    CanExtractMotifs,
    # Similarity protocols
    CanComputeSimilarity,
    CanComputeMotifSimilarity,
    CanComputeSpectralSimilarity,
    # Corpus protocols
    CanManageCorpus,
    CanQueryCorpus,
    # Resonance protocols
    CanFindResonances,
    CanRankResonances,
    # Interpretation protocols
    CanExplainResonance,
    CanExplainMotifs,
    CanGenerateReports,
    # Composite protocols
    FullSimilarityService,
    FullCorpusService,
    FullInterpretationService,
)

# Service container (dependency injection)
from .container import (
    ServiceContainer,
    create_service_container,
    create_test_container,
    get_container,
    reset_container,
    set_container,
)

# Graph Analogies (novel research direction)
from .graph_analogies import (
    GraphEmbedding,
    AnalogyResult,
    StructuralEncoder,
    GraphAnalogyEngine,
)

# Game Time Translation (discrete→continuous market→game)
from .game_time_translation import (
    TimeMode,
    Candlestick,
    GameMoment,
    GameSegment,
    DiscreteToGameTranslator,
)

# Cave Trader (market structure as navigable terrain)
from .cave_trader import (
    OHLCV,
    CaveSlice,
    RocketState,
    MusicState,
    CaveGenerator,
    MusicGenerator,
    CaveTraderGame,
)

# Market Instruments (harmonic sonification of market data)
from .market_instruments import (
    PositionType,
    Position,
    TradingAccount,
    TradingMechanics,
    Harmonic,
    InstrumentState,
    MarketInstrument,
    VolumeInstrument,
    VolatilityInstrument,
    ChordVoicing,
    MarketOrchestra,
)

# Market Harmonics (true cycle detection and spectral analysis)
from .market_harmonics import (
    DetectedCycle,
    MarketSpectrum,
    CycleDetector,
    SpectralSynthesizer,
)

# Pattern Loop (chord progressions and visual projections for player turns)
from .pattern_loop import (
    Chord,
    ChordProgression,
    ChordBuilder,
    CaveProjector,
    CaveSegment,
    PatternProjection,
    TurnPhase,
    TurnState,
    TurnManager,
)

# Market Data Sources (free APIs for testing)
from .market_data_sources import (
    Bar,
    YahooFinanceSource,
    BinanceSource,
    AlphaVantageSource,
    SampleDataSource,
    save_bars,
    load_bars,
)

# Cave Instruments (2D flight-style instrument panel)
from .cave_instruments import (
    ThreatLevel,
    VerticalReading,
    HorizontalReading,
    PositionReading,
    InstrumentPanel,
    InstrumentCalculator,
    ASCIIRenderer,
)

# Binance Live Data (real-time streaming)
from .binance_live import (
    StreamType,
    LiveTick,
    LiveBar,
    OrderBookSnapshot,
    SimulatedPosition,
    SimulatedAccount,
    BinanceLiveStream,
    LiveCaveGame,
)

# Game Menu (kid-friendly crypto selection)
from .game_menu import (
    CryptoChoice,
    CaveOption,
    CAVE_OPTIONS,
    GameMenu,
    render_menu,
    render_loading,
    render_controls,
    render_game_over,
    get_symbol_from_choice,
)

# Egg Trader (position-based trading game)
from .egg_trader import (
    EggType,
    Egg,
    EggTraderGame,
    Portfolio,
)

# Egg Trader v2 (position sizing and order book)
from .egg_trader_v2 import (
    EggTraderV2,
    EggPile,
    OrderBookLevel,
    CarriedEggs,
)

# Egg Trading Floor (full order book simulation)
from .egg_trading_floor import (
    Side,
    PriceLevel,
    Position as TradingPosition,
    LimitOrder,
    Trade,
    Portfolio as TradingPortfolio,
    EggTradingFloor,
)

# Egg Trader Levels (progressive difficulty)
from .egg_trader_levels import (
    EggColor,
    ZoneType,
    EggZone,
    CarriedPosition,
    CompletedTrade,
    TradingFloor,
    DifficultyLevel,
    EggTraderLevels,
    # Support/Resistance
    LevelType,
    PriceLevel as SRPriceLevel,
    PriceBar,
)

# Simple Cave (Nokia-era simplicity game)
from .simple_cave import (
    GameState,
    CaveSlice as SimpleCaveSlice,
    Rocket,
    Game,
    SimpleCaveGame,
    HistoryTracker,
    render_vertical_history,
    render_horizontal_history,
)

# Snake Trail (movement history visualization)
from .snake_trail import (
    Direction,
    SnakeTrail,
    CaveWithSnake,
)

# Game Levels (historical data as playable levels)
from .game_levels import (
    LevelBar,
    GameLevel,
    SnakePosition,
    SnakeAnalysis,
    LevelGenerator,
    SnakeTracker,
)

# Snake Music (2D position to musical space - foreshadowing)
from .snake_music import (
    ChordQuality,
    MusicalNote,
    MusicalChord,
    MusicalPhrase,
    SnakeToMusic,
    SnakeSynth,
)

# Price Harmonics (market structure as musical harmonics)
from .price_harmonics import (
    SwingType,
    SwingPoint,
    PriceChord,
    HarmonicSequence,
    SwingDetector,
    AmplitudeScaler,
    PriceHarmonicsEngine,
    piano_key_to_note_name,
)

# Market Replay Client (recorded data playback for backtesting)
from .market_replay import (
    MarketReplayClient,
    SyncMarketReplayClient,
    ReplayFrame,
    RecordingInfo,
    OrderBookData as ReplayOrderBook,
    TradeData as ReplayTrade,
    CanStreamReplay,
    CanGetBatch,
    CanListRecordings,
)

__version__ = "0.16.0"
__all__ = [
    # Core signature types
    "StructuralSignature",
    "SpectralSignature",
    "Resonance",

    # Extractors
    "SignatureExtractor",
    "SpectralExtractor",

    # Motif detection
    "MotifDetector",
    "spectral_similarity",

    # Pruning
    "GraphPruner",
    "auto_prune",
    "PruningStats",

    # Similarity computation (NEW)
    "SimilarityService",
    "SimilarityResult",
    "compute_similarity",

    # Corpus management (NEW)
    "Corpus",
    "CorpusService",
    "create_corpus",
    "load_corpus",
    "save_corpus",

    # Resonance finding (NEW)
    "ResonanceService",
    "find_resonances",
    "find_cross_domain_resonances",

    # Interpretation (NEW)
    "InterpretationService",
    "explain_resonance",
    "generate_comparison_report",

    # Domain adapters
    "DomainAdapter",
    "TextAdapter",
    "ImageAdapter",
    "MusicAdapter",
    "CodeAdapter",
    "get_adapter",
    "adapt_to_graph",
    "validate_graph",
    "GraphValidationResult",

    # Protocol definitions (interfaces)
    "SignatureData",
    "ResonanceData",
    "SimilarityData",
    "CanAdaptToGraph",
    "ProvidesGraphValidation",
    "CanExtractSignatures",
    "CanExtractMotifs",
    "CanComputeSimilarity",
    "CanComputeMotifSimilarity",
    "CanComputeSpectralSimilarity",
    "CanManageCorpus",
    "CanQueryCorpus",
    "CanFindResonances",
    "CanRankResonances",
    "CanExplainResonance",
    "CanExplainMotifs",
    "CanGenerateReports",
    "FullSimilarityService",
    "FullCorpusService",
    "FullInterpretationService",

    # Dependency injection container
    "ServiceContainer",
    "create_service_container",
    "create_test_container",
    "get_container",
    "reset_container",
    "set_container",

    # Graph Analogies (novel research)
    "GraphEmbedding",
    "AnalogyResult",
    "StructuralEncoder",
    "GraphAnalogyEngine",

    # Game Time Translation (market→game)
    "TimeMode",
    "Candlestick",
    "GameMoment",
    "GameSegment",
    "DiscreteToGameTranslator",

    # Cave Trader (market as navigable terrain)
    "OHLCV",
    "CaveSlice",
    "RocketState",
    "MusicState",
    "CaveGenerator",
    "MusicGenerator",
    "CaveTraderGame",

    # Market Instruments (harmonic sonification)
    "PositionType",
    "Position",
    "TradingAccount",
    "TradingMechanics",
    "Harmonic",
    "InstrumentState",
    "MarketInstrument",
    "VolumeInstrument",
    "VolatilityInstrument",
    "ChordVoicing",
    "MarketOrchestra",

    # Market Harmonics (true cycle detection)
    "DetectedCycle",
    "MarketSpectrum",
    "CycleDetector",
    "SpectralSynthesizer",

    # Pattern Loop (chord progressions and turn management)
    "Chord",
    "ChordProgression",
    "ChordBuilder",
    "CaveProjector",
    "CaveSegment",
    "PatternProjection",
    "TurnPhase",
    "TurnState",
    "TurnManager",

    # Market Data Sources (free APIs for testing)
    "Bar",
    "YahooFinanceSource",
    "BinanceSource",
    "AlphaVantageSource",
    "SampleDataSource",
    "save_bars",
    "load_bars",

    # Cave Instruments (2D flight-style gauges)
    "ThreatLevel",
    "VerticalReading",
    "HorizontalReading",
    "PositionReading",
    "InstrumentPanel",
    "InstrumentCalculator",
    "ASCIIRenderer",

    # Simple Cave (Nokia-era game)
    "GameState",
    "SimpleCaveSlice",
    "Rocket",
    "Game",
    "SimpleCaveGame",
    "HistoryTracker",
    "render_vertical_history",
    "render_horizontal_history",

    # Snake Trail (movement history)
    "Direction",
    "SnakeTrail",
    "CaveWithSnake",

    # Game Levels (historical data as levels)
    "LevelBar",
    "GameLevel",
    "SnakePosition",
    "SnakeAnalysis",
    "LevelGenerator",
    "SnakeTracker",

    # Snake Music (2D position to musical space)
    "ChordQuality",
    "MusicalNote",
    "MusicalChord",
    "MusicalPhrase",
    "SnakeToMusic",
    "SnakeSynth",

    # Price Harmonics (market structure as musical harmonics)
    "SwingType",
    "SwingPoint",
    "PriceChord",
    "HarmonicSequence",
    "SwingDetector",
    "AmplitudeScaler",
    "PriceHarmonicsEngine",
    "piano_key_to_note_name",

    # Binance Live Data (real-time streaming)
    "StreamType",
    "LiveTick",
    "LiveBar",
    "OrderBookSnapshot",
    "SimulatedPosition",
    "SimulatedAccount",
    "BinanceLiveStream",
    "LiveCaveGame",

    # Game Menu (kid-friendly crypto selection)
    "CryptoChoice",
    "CaveOption",
    "CAVE_OPTIONS",
    "GameMenu",
    "render_menu",
    "render_loading",
    "render_controls",
    "render_game_over",
    "get_symbol_from_choice",

    # Egg Trader (position-based trading game)
    "EggType",
    "Egg",
    "EggTraderGame",
    "Portfolio",

    # Egg Trader v2 (position sizing and order book)
    "EggTraderV2",
    "EggPile",
    "OrderBookLevel",
    "CarriedEggs",

    # Egg Trading Floor (full order book simulation)
    "Side",
    "PriceLevel",
    "TradingPosition",
    "LimitOrder",
    "Trade",
    "TradingPortfolio",
    "EggTradingFloor",

    # Egg Trader Levels (progressive difficulty)
    "EggColor",
    "ZoneType",
    "EggZone",
    "CarriedPosition",
    "CompletedTrade",
    "TradingFloor",
    "DifficultyLevel",
    "EggTraderLevels",
    # Support/Resistance
    "LevelType",
    "SRPriceLevel",
    "PriceBar",

    # Market Replay Client (recorded data playback)
    "MarketReplayClient",
    "SyncMarketReplayClient",
    "ReplayFrame",
    "RecordingInfo",
    "ReplayOrderBook",
    "ReplayTrade",
    "CanStreamReplay",
    "CanGetBatch",
    "CanListRecordings",
]
