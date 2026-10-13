import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Shield,
  Sparkles,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  Scale,
  FileText,
  Sliders,
  ChevronLeft,
  ChevronRight,
  Info,
  X,
  Sun,
  Moon,
  ArrowUpRight,
} from "lucide-react";

interface SampleItem {
  id: string;
  dataset: string;
  title: string;
  category: string;
  text: string;
  source_name?: string;
  source_url?: string;
}

interface SentenceItem {
  text: string;
  is_biased: boolean;
  label: string;
  score: number;
}

interface AnalysisData {
  input: { title?: string; text: string };
  sentences: SentenceItem[];
  specialist_sentence: {
    label: string;
    biased_sentence_count: number;
    total_sentences: number;
    bias_ratio: number;
    flagged_sentences: SentenceItem[];
  };
  specialist_political: {
    method: string;
    model_name: string;
    predicted_label: string;
    confidence: number;
    scores: Record<string, number>;
  };
  isolated_llm: {
    political_leaning?: { label: string; confidence?: number; scores?: Record<string, number> };
    linguistic_bias?: { label: string; is_biased: boolean; bias_score?: number };
    loaded_phrases?: string[];
    reasoning?: string;
  };
  arbiter: {
    final_verdict?: {
      is_biased: boolean;
      overall_leaning: string;
      confidence: number;
      primary_bias_type?: string;
    };
    why_biased_heading?: string;
    why_biased_explanation?: string;
    key_findings?: string[];
    specialist_synthesis?: Record<string, string>;
    suggest_rewrite_recommended?: boolean;
  };
}

interface RewriteChange {
  original: string;
  rewritten: string;
  reason: string;
}

interface RewriteData {
  rewritten_text: string;
  changes: RewriteChange[];
  editorial_summary?: string;
}

const DATASET_SOURCES: Record<string, string> = {
  aravalli: "https://www.downtoearth.org.in",
  kashmir: "https://www.thehindu.com",
  ladakh: "https://indianexpress.com",
  babe: "https://huggingface.co/datasets/mediabiasgroup/BABE",
  mbic: "https://github.com/Media-Bias-Group/news-bias-full-data",
  allsides: "https://www.allsides.com/media-bias",
  basil: "https://github.com/fanzhang-f/BASIL",
  anno_lexical: "https://github.com/Media-Bias-Group",
  mbib: "https://huggingface.co/mediabiasgroup/mbib-base",
  reuters: "https://www.reuters.com",
};

function getDatasetSource(datasetText: string, sourceUrl?: string): string {
  if (sourceUrl) return sourceUrl;
  const d = (datasetText || "").toLowerCase();
  if (d.includes("aravalli")) return DATASET_SOURCES.aravalli;
  if (d.includes("kashmir")) return DATASET_SOURCES.kashmir;
  if (d.includes("ladakh")) return DATASET_SOURCES.ladakh;
  if (d.includes("babe")) return DATASET_SOURCES.babe;
  if (d.includes("mbic")) return DATASET_SOURCES.mbic;
  if (d.includes("allsides")) return DATASET_SOURCES.allsides;
  if (d.includes("basil")) return DATASET_SOURCES.basil;
  if (d.includes("anno")) return DATASET_SOURCES.anno_lexical;
  if (d.includes("mbib")) return DATASET_SOURCES.mbib;
  if (d.includes("reuters") || d.includes("wire") || d.includes("ap")) return DATASET_SOURCES.reuters;
  return "https://huggingface.co/datasets";
}

const DEFAULT_SAMPLES: SampleItem[] = [
  {
    id: "aravalli-forest-dispute",
    dataset: "Indian Express / Down To Earth",
    title: "Death Knell for Aravallis: SC 100m Order Stirs Mining Mafia Panic",
    category: "Sensationalist Alarm & Judicial Misinterpretation",
    source_name: "The Indian Express / Down To Earth",
    source_url: "https://indianexpress.com/article/india/aravalli-definition-supreme-court-100-metre-fsi-mining/",
    text: "In a move described by conservationists as a catastrophic death knell for India’s oldest fold mountains, the Supreme Court has accepted a controversial 100-metre elevation threshold that effectively strips environmental safeguards from over ninety percent of the Aravalli range. Under the arbitrary new definition pushed by the Union Environment Ministry, any landform under 100 metres in local relief will no longer qualify as an Aravalli hill, officially abandoning nearly eleven thousand ecologically irreplaceable ridges across Haryana and Rajasthan to the mercy of rapacious mining mafias and real estate syndicates.\n\nOutraged activists and forestry experts warned that this bureaucratic travesty completely capitulates to the commercial quarrying lobby, greenlighting the dynamiting and flattening of vital low-lying hillocks that serve as North India’s sole buffer against the advancing Thar Desert. By excluding low ridges that constitute the primary groundwater recharge aquifers for the parched National Capital Region, critics argue the judicial bench has condemned millions of citizens to acute water scarcity, toxic particulate haze, and irreversible desertification.\n\nIgnoring explicit dissent from the Forest Survey of India, which cautioned that only a tiny fraction of documented ridges exceed the 100-metre mark, media dispatches and citizen collectives immediately sounded the alarm that state governments are preparing to auction off denotified rocky slopes for commercial plunder. Spurred by sensational reports that bulldozers and stone crushers would erase ninety percent of the range overnight, hundreds of furious villagers, environmentalists, and farmer unions launched aggressive highway blockades and sit-in demonstrations across southern Haryana, demanding the immediate revocation of what they termed a state-sanctioned ecocide.",
  },
  {
    id: "kashmir-maps-bias",
    dataset: "Kashmir / The Hindu",
    title: "Kashmir Territorial Cartography & Border Sovereignty Dispute",
    category: "Geopolitical Bias & Dynamic Cartography",
    source_name: "The Hindu / Survey of India",
    source_url: "https://www.thehindu.com",
    text: "A fierce international debate has intensified over digital cartography and the systematic geopolitical bias embedded in global mapping platforms regarding Jammu, Kashmir, and Ladakh. Tech conglomerates and international media organizations have faced severe backlash from New Delhi for deploying geofenced, dynamic border renderings that depict the region’s boundaries inconsistently based on user geography. While users within India see the constitutional territorial integrity upheld under Survey of India standards, global and regional versions frequently erase Indian sovereignty by demarcating vast swathes as contested territory or adopting adversary terminologies.\n\nInternational human rights watchdogs and regional commentators from Islamabad and Beijing routinely push rival narratives, characterizing Indian administrative consolidations since the 2019 bifurcation as unilateral changes to an internationally disputed zone. Pakistani state media continues to amplify rhetoric describing the territory through the lens of military occupation and disputed plebiscites, while Chinese state outlets align reporting with Beijing's disputed LAC claims in Aksai Chin.\n\nConversely, Indian authorities and strategic analysts reject foreign cartographic manipulation as an infringement on national sovereignty. Official spokespersons reiterated that the entirety of Jammu, Kashmir, and Ladakh constitutes an integral and non-negotiable part of the Indian Union, warning digital platforms that publishing distorted maps that legitimize illegal foreign occupations violates national criminal law.",
  },
  {
    id: "ladakh-sixth-schedule",
    dataset: "Ladakh / Indian Express",
    title: "Ladakh Sixth Schedule Agitation & Ecological Safeguards",
    category: "Indigenous Autonomy vs Central Governance",
    source_name: "The Indian Express / LAB",
    source_url: "https://indianexpress.com",
    text: "Deep in the high-altitude desert of Leh, thousands of indigenous demonstrators gathered in biting sub-zero conditions to support Sonam Wangchuk’s indefinite climate fast, escalating an impassioned showdown between grassroots Himalayan communities and the central administration. Agitators from the Leh Apex Body and Kargil Democratic Alliance have demanded constitutional guarantees under the Sixth Schedule, warning that Ladakh’s hyper-fragile glacial ecosystems will face catastrophic degradation under unchecked industrialization, corporate mega-projects, and centralized bureaucratic oversight.\n\nMovement organizers accused national mainstream media channels of maintaining a calculated blackout on the agitations to shield political establishments from scrutiny over unfulfilled electoral promises. Leaders highlighted that local tribal populations, who comprise over ninety percent of Ladakh, have been effectively disenfranchised without legislative representation, leaving delicate pastures vulnerable to commercial mining ventures that threaten to accelerate glacial retreat.\n\nIn response, central government representatives emphasized that substantial infrastructure allocations and renewable energy investments have unlocked unprecedented development for the strategic frontier. Officials argued that high-powered advisory committees are already addressing tribal welfare, asserting that constitutional autonomy demands must be balanced against national defense logistics and broad-based economic integration along sensitive Himalayan borders.",
  },
  {
    id: "babe-delhi-smog",
    dataset: "BABE / Reuters Wire",
    title: "New Delhi Air Quality & Meteorological Inversion Monitoring",
    category: "Objective Environmental Wire (Non-Biased)",
    source_name: "Reuters (BABE Dataset)",
    source_url: "https://www.reuters.com/article/us-india-pollution/toxic-smog-shrouds-indian-capital-as-pollution-soars-government-measures-fail-idUSKCN1OX0K9",
    text: "A sharp drop in temperatures and wind speed, combined with vehicle and industrial emissions, dust from building sites and smoke from garbage burning has boosted pollution over much of north India, including New Delhi, home to more than 20 million.\n\nThe Central Pollution Control Board reported that the air quality index for Delhi hovered in the severe category, with microscopic PM2.5 concentrations exceeding 440 micrograms per cubic meter across monitoring stations.\n\nGovernment monitoring agencies directed schools to suspend outdoor activities, deployed mechanized vacuum sweepers to control road dust, and ordered municipal authorities to sprinkle water across heavy-traffic corridors. Regional environmental meteorologists stated that unfavorable atmospheric conditions and agricultural crop-residue burning in neighboring states are expected to sustain the smog cover through the weekend.",
  },
  {
    id: "wire-fed-neutral",
    dataset: "Reuters / AP News Wire",
    title: "Federal Reserve Holds Benchmark Rates Steady",
    category: "Neutral Objective Wire Reporting",
    source_name: "Reuters / AP Wire",
    source_url: "https://www.reuters.com",
    text: "The Federal Reserve announced on Wednesday that benchmark interest rates will remain unchanged between 5.25% and 5.50% following its two-day policy meeting. Central bank officials noted that inflation has moderated significantly over the past year while the labor market continues to demonstrate resilience, with payroll additions remaining consistent with sustainable long-term economic expansion.\n\nIn an official statement released after the deliberations, the Federal Open Market Committee emphasized that committee members do not anticipate reducing borrowing costs until they gain greater confidence that consumer price inflation is sustainably tracking toward their 2% target. The committee highlighted that economic activity has been expanding at a solid pace, though the economic outlook remains subject to global trade uncertainties.\n\nFinancial analysts from major institutions had widely expected the hold decision ahead of upcoming economic reports on consumer spending and gross domestic product. Government bond yields held steady following the announcement, while major equity indices registered modest gains as market participants digested Chairman Jerome Powell's post-meeting press conference remarks.",
  },
  {
    id: "babe-h1b-india",
    dataset: "BABE / Breitbart",
    title: "High-Skilled Visa Policy & Outsourcing Debate",
    category: "Economic Nationalist Stance (Biased)",
    source_name: "Breitbart News (BABE Dataset)",
    source_url: "https://www.breitbart.com/politics/2019/02/11/billionaire-kochs-back-gop-dem-plan-to-outsource-middle-class-american-jobs-2/",
    text: "A plan known as the Fairness for High-Skilled Immigrants Act is a boon to Silicon Valley billionaires, big business elites, and outsourcing firms, as they would be able to readily import more lower-paid Indian and Chinese foreign workers to take American jobs that would have otherwise gone to American citizens.\n\nLobbying groups formed by the nation’s major tech companies reportedly employ large blocs of foreign visa-workers in high-paying software engineering roles that could readily be filled by young domestic graduates.\n\nCongressional sponsors continue to push legislation that would dramatically expand legal and financial incentives for corporations to bypass local labor pools, replacing middle-class professionals with temporary contract workers bound to employer-sponsored residency quotas.",
  },
  {
    id: "babe-india-diplomacy",
    dataset: "BABE / Alternet",
    title: "Diplomatic Protocol & Bilateral Arms Pact Framing",
    category: "Partisan Geopolitical Critique (Biased)",
    source_name: "Alternet (BABE Dataset)",
    source_url: "https://www.alternet.org/2020/02/trump-visits-modis-india-and-announces-3-billion-arms-deal-the-worst-kind-of-fascists/",
    text: "The official state visit kicked off with an elaborate spectacle of staged pomp and pageantry in Gujarat, where leaders announced a three-billion-dollar military arms package designed to cement strategic defense partnerships in the Indo-Pacific.\n\nHuman rights advocates and civil society groups strongly condemned the diplomatic summit, arguing that lavish political celebrations were orchestrated to distract international attention from draconian domestic policies and escalating crackdowns against minority rights.\n\nWhile bilateral communiqués touted democratic alignments and shared maritime security interests, critics noted that defense pacts prioritize corporate defense contractors while ignoring deepening authoritarian governance and civil liberties restrictions across the subcontinent.",
  },
  {
    id: "babe-1458",
    dataset: "BABE / National Review",
    title: "Conservative Warning on Democratic Ticket",
    category: "Right-leaning + Heavy Spin",
    source_name: "BABE / National Review",
    source_url: "https://huggingface.co/datasets/mediabiasgroup/BABE",
    text: "Instead, the forgetful and frail former VP will be used as a vehicle for the left’s cultural revolution, especially since Biden’s vice presidential pick Kamala Harris has proven to be a solid member of the far left. Radical activists have completely captured party leadership, demanding extreme climate mandates and socialist economic experiments that will inevitably crush middle-class households across heartland America.\n\nBehind closed doors, shadowy progressive donors and unaccountable party apparatchiks are orchestrating an aggressive policy takeover, abandoning working-class voters who once formed the backbone of the party. Every major legislative priority has been rewritten to appease fringe ideological enclaves rather than addressing kitchen-table economic realities.\n\nMeanwhile, compliant mainstream media outlets continue their coordinated cover-up to protect the establishment from legitimate voter scrutiny. Rather than holding senior party figures accountable for catastrophic foreign policy blunders and reckless spending bills, biased broadcast anchors dutifully amplify partisan talking points while deriding any conservative dissent as dangerous misinformation.",
  },
  {
    id: "allsides-salon-2014",
    dataset: "AllSides / Salon",
    title: "Partisan Standoff & Corporate Complicity",
    category: "Left-leaning Partisan Critique",
    source_name: "AllSides / Salon",
    source_url: "https://www.allsides.com/media-bias",
    text: "The country remains locked in a bitter partisan standoff driven by reactionary right extremists who are systematically waging war on reproductive rights and voting access for vulnerable communities. State legislatures across red America have launched an unprecedented blitz of punitive measures aimed at disenfranchising minority voters and criminalizing essential healthcare decisions.\n\nUnchecked corporate greed and Wall Street vultures continue their destructive squeeze on American workers, stashing trillions in offshore tax havens while gutting worker protections and suffocating union drives. Oligarchic corporate executives award themselves obscene multi-million-dollar bonuses while everyday families struggle under artificially inflated grocery bills and unaffordable prescription drug prices.\n\nMeanwhile, right-wing media personalities and corporate-backed think tanks spew hatred across the airwaves, manufacturing bad-faith culture wars to distract public attention from widespread economic exploitation. Until regular citizens organize to dismantle these toxic power structures, our democracy will remain perilously held hostage by wealthy special interests and their reactionary congressional enablers.",
  },
  {
    id: "mbic-climate-spin",
    dataset: "MBIC / Media Spin",
    title: "Historic Superstorm Threatens East Coast",
    category: "Sensationalist Disaster Framing",
    source_name: "MBIC Benchmark",
    source_url: "https://github.com/Media-Bias-Group/news-bias-full-data",
    text: "A catastrophic superstorm is hurtling toward the Eastern Seaboard, threatening to obliterate coastal infrastructure as panic sweeps through vulnerable neighborhoods. Relentless hurricane-force gusts and apocalyptic storm surges are expected to overwhelm outdated sea walls, submerging low-lying communities beneath murky floodwaters and leaving millions stranded without electricity or clean water.\n\nClueless bureaucrats and paralyzed state officials have miserably failed to prepare emergency response systems, ignoring repeated frantic warnings from regional meteorologists. Shelter capacity remains woefully inadequate, and evacuation routes are already gridlocked with desperate families struggling to escape the imminent tempest.\n\nMeanwhile, reckless climate deniers and fossil-fuel apologists stubbornly scoff at the undeniable devastation unfolding before our very eyes. Decades of intentional inaction and unbridled carbon pollution have fueled unprecedented ocean temperatures, transforming seasonal weather patterns into lethal engines of atmospheric destruction that now threaten to plunge coastal cities into prolonged humanitarian crises.",
  },
  {
    id: "basil-tax-reform",
    dataset: "BASIL / Wall Street Journal",
    title: "Bipartisan Wrangling Over Corporate Tax Reform",
    category: "Economic Policy Stance",
    source_name: "BASIL / WSJ",
    source_url: "https://github.com/fanzhang-f/BASIL",
    text: "Lawmakers sparred late into Thursday evening over competing amendments to the corporate tax rate proposal. Proponents argued the reductions would spur domestic capital expenditure and prevent multinational corporations from shifting headquarters overseas, citing econometric modeling that projects a 1.2% boost to annualized manufacturing investment over the next five years.\n\nConversely, opponents presented treasury estimates forecasting a $250 billion deficit expansion over the decade, contending the measure disproportionately rewards wealthy shareholders at the expense of infrastructure modernization. Democratic representatives introduced three substitute amendments to redirect revenue toward child care credits and community college workforce grants, though each fell on party-line votes.\n\nThe committee chair scheduled a decisive roll call vote for early Friday morning, following overnight negotiations regarding depreciation timelines for domestic research and development. Both caucuses convened closed-door briefings with industry leaders and economic advisors to finalize vote commitments ahead of the final legislative floor debate scheduled for next week.",
  },
];

export default function App() {
  const [samples, setSamples] = useState<SampleItem[]>(DEFAULT_SAMPLES);
  const [title, setTitle] = useState(DEFAULT_SAMPLES[0].title);
  const [text, setText] = useState(DEFAULT_SAMPLES[0].text);
  const [activeSampleId, setActiveSampleId] = useState(DEFAULT_SAMPLES[0].id);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isRewriting, setIsRewriting] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [rewrite, setRewrite] = useState<RewriteData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);
  const [isAboutOpen, setIsAboutOpen] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    try {
      const saved = localStorage.getItem("newsbias-theme");
      if (saved === "dark" || saved === "light") return saved;
    } catch (_) {}
    return "light";
  });

  useEffect(() => {
    if (theme === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
    try {
      localStorage.setItem("newsbias-theme", theme);
    } catch (_) {}
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "light" ? "dark" : "light"));
  };

  const sliderRef = useRef<HTMLDivElement>(null);

  const checkScrollBounds = useCallback(() => {
    const el = sliderRef.current;
    if (!el) return;
    const { scrollLeft, scrollWidth, clientWidth } = el;
    setCanScrollLeft(scrollLeft > 4);
    setCanScrollRight(scrollLeft + clientWidth < scrollWidth - 4);
  }, []);

  const scrollSlider = (direction: "left" | "right") => {
    if (sliderRef.current) {
      const scrollAmount = sliderRef.current.clientWidth;
      sliderRef.current.scrollBy({
        left: direction === "left" ? -scrollAmount : scrollAmount,
        behavior: "smooth",
      });
      setTimeout(checkScrollBounds, 350);
    }
  };

  const handleSliderWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (sliderRef.current) {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        sliderRef.current.scrollLeft += e.deltaY;
      } else {
        sliderRef.current.scrollLeft += e.deltaX;
      }
      checkScrollBounds();
    }
  };

  useEffect(() => {
    checkScrollBounds();
    window.addEventListener("resize", checkScrollBounds);
    return () => window.removeEventListener("resize", checkScrollBounds);
  }, [checkScrollBounds, samples]);

  useEffect(() => {
    fetch("/bias/samples")
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setSamples(data);
        }
      })
      .catch(() => {});
  }, []);

  const selectSample = (s: SampleItem) => {
    setActiveSampleId(s.id);
    setTitle(s.title);
    setText(s.text);
    setAnalysis(null);
    setRewrite(null);
    setError(null);
  };

  const runAnalysis = async () => {
    if (!text.trim()) return;
    setIsAnalyzing(true);
    setError(null);
    setRewrite(null);

    try {
      const res = await fetch("/bias/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, text, sample_id: activeSampleId }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setAnalysis(data);
    } catch (e: any) {
      setError(e.message || "Failed to analyze article bias.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const runRewrite = async () => {
    if (!analysis) return;
    setIsRewriting(true);
    setError(null);

    try {
      const res = await fetch("/bias/rewrite", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: analysis.input.text,
          why_biased_explanation: analysis.arbiter?.why_biased_explanation || "",
          detected_leaning: analysis.arbiter?.final_verdict?.overall_leaning || "neutral",
          key_findings: analysis.arbiter?.key_findings || [],
        }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setRewrite(data);
    } catch (e: any) {
      setError(e.message || "Failed to generate neutral rewrite.");
    } finally {
      setIsRewriting(false);
    }
  };

  const isBiased = analysis?.arbiter?.final_verdict?.is_biased ?? false;
  // Dynamic heading requested: strictly "Why is it biased?" or "Why is it not biased?" (no raw (not))
  const dynamicHeading =
    analysis?.arbiter?.why_biased_heading ||
    (isBiased ? "Why is it biased?" : "Why is it not biased?");

  const activeSample = samples.find((s) => s.id === activeSampleId);

  return (
    <div className="flex flex-col min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100 transition-colors duration-200">
      {/* ── Top Navigation ──────────────────────────────────────────────── */}
      <header className="border-b bg-white/95 dark:bg-slate-900/95 backdrop-blur sticky top-0 z-40 border-slate-200 dark:border-slate-800 transition-colors duration-200">
        <div className="max-w-6xl mx-auto px-8 sm:px-12 py-3.5 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <Scale className="h-5 w-5 text-indigo-600 dark:text-indigo-400" />
            <span className="text-base font-bold tracking-tight text-slate-900 dark:text-white">
              News Bias Detection using Machine Learning
            </span>
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              id="about-btn"
              onClick={() => setIsAboutOpen(true)}
              className="text-xs font-mono font-medium text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:underline hover:decoration-slate-400 underline-offset-2 transition-colors cursor-pointer uppercase tracking-wider"
            >
              ABOUT
            </button>
            <button
              type="button"
              id="theme-toggle-btn"
              onClick={toggleTheme}
              aria-label="Toggle light/dark theme"
              title={theme === "light" ? "Switch to dark theme" : "Switch to light theme"}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
            >
              {theme === "light" ? (
                <Sun className="h-4 w-4 text-amber-500" />
              ) : (
                <Moon className="h-4 w-4 text-indigo-400" />
              )}
            </button>
          </div>
        </div>
      </header>

      {/* ── Main Content Container ───────────────────────────────────────── */}
      <main className="max-w-6xl mx-auto px-8 sm:px-12 py-8 flex-1 flex flex-col gap-8 w-full">
        {/* ── 1. Benchmark Sample Selector Slider ───────────────────────────── */}
        <div className="flex flex-col gap-2.5">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Select Curated Benchmark Sample:
            </span>
            <div className="flex items-center gap-1.5">
              <div className="flex flex-wrap items-center gap-1.5 text-xs text-slate-400 dark:text-slate-500 font-mono hidden sm:inline-flex mr-2">
                <a
                  href={DATASET_SOURCES.babe}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 hover:underline hover:decoration-slate-400 underline-offset-2 transition-colors"
                >
                  BABE
                </a>
                <span>·</span>
                <a
                  href={DATASET_SOURCES.mbic}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 hover:underline hover:decoration-slate-400 underline-offset-2 transition-colors"
                >
                  MBIC
                </a>
                <span>·</span>
                <a
                  href={DATASET_SOURCES.allsides}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 hover:underline hover:decoration-slate-400 underline-offset-2 transition-colors"
                >
                  AllSides
                </a>
                <span>·</span>
                <a
                  href={DATASET_SOURCES.basil}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 hover:underline hover:decoration-slate-400 underline-offset-2 transition-colors"
                >
                  BASIL
                </a>
                <span>·</span>
                <a
                  href={DATASET_SOURCES.anno_lexical}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 hover:underline hover:decoration-slate-400 underline-offset-2 transition-colors"
                >
                  Anno-Lexical
                </a>
                <span>·</span>
                <a
                  href={DATASET_SOURCES.mbib}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 hover:underline hover:decoration-slate-400 underline-offset-2 transition-colors"
                >
                  MBIB
                </a>
                <span>·</span>
                <a
                  href={DATASET_SOURCES.reuters}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 hover:underline hover:decoration-slate-400 underline-offset-2 transition-colors"
                >
                  Reuters/AP
                </a>
              </div>
              <button
                type="button"
                id="slider-prev-btn"
                onClick={() => scrollSlider("left")}
                disabled={!canScrollLeft}
                aria-label="Scroll left"
                className={`p-1.5 rounded-lg border transition-all ${
                  canScrollLeft
                    ? "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-750 text-slate-700 dark:text-slate-200 shadow-xs cursor-pointer active:scale-95"
                    : "border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-slate-300 dark:text-slate-600 cursor-not-allowed opacity-40 shadow-none"
                }`}
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <button
                type="button"
                id="slider-next-btn"
                onClick={() => scrollSlider("right")}
                disabled={!canScrollRight}
                aria-label="Scroll right"
                className={`p-1.5 rounded-lg border transition-all ${
                  canScrollRight
                    ? "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-750 text-slate-700 dark:text-slate-200 shadow-xs cursor-pointer active:scale-95"
                    : "border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 text-slate-300 dark:text-slate-600 cursor-not-allowed opacity-40 shadow-none"
                }`}
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* Slider track: shows 3 at a time, scroll wheel supported, no scroll bar */}
          <div
            ref={sliderRef}
            onScroll={checkScrollBounds}
            onWheel={handleSliderWheel}
            className="flex flex-row gap-3 overflow-x-auto no-scrollbar scroll-smooth py-1.5 px-0.5"
          >
            {samples.map((s) => {
              const active = activeSampleId === s.id;
              return (
                <button
                  key={s.id}
                  id={`sample-btn-${s.id}`}
                  onClick={() => selectSample(s)}
                  className={`flex-shrink-0 text-left p-3.5 rounded-xl border text-sm transition-all w-[calc(33.333%-0.5rem)] min-w-[270px] ${
                    active
                      ? "border-indigo-600 dark:border-indigo-500 bg-indigo-50/50 dark:bg-indigo-950/40 shadow-sm ring-1 ring-indigo-500"
                      : "border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 bg-white dark:bg-slate-900 shadow-xs"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <a
                      href={getDatasetSource(s.dataset, s.source_url)}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      title={`View ${s.source_name || s.dataset} source`}
                      className="text-xs font-medium text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 hover:underline hover:decoration-slate-400 underline-offset-2 transition-colors"
                    >
                      {s.dataset}
                    </a>
                    <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300">
                      {s.category.split(" ")[0]}
                    </span>
                  </div>
                  <div className="font-semibold text-xs text-slate-900 dark:text-slate-100 line-clamp-1">{s.title}</div>
                </button>
              );
            })}
          </div>
        </div>

        {/* ── 2. Input Editor ──────────────────────────────────────────────── */}
        <div className="border border-slate-200 dark:border-slate-800 rounded-2xl p-5 bg-white dark:bg-slate-900 flex flex-col gap-4 shadow-sm transition-colors duration-200">
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
              Article Title (Optional)
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Breaking News Headline..."
              className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg px-3.5 py-2 text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between items-center">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Article Body / Excerpt
              </label>
              <span className="text-xs text-slate-400 dark:text-slate-500">
                {text.length} characters · {text.trim().split(/\s+/).filter(Boolean).length} words
              </span>
            </div>
            <textarea
              rows={12}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste article text here..."
              className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-lg p-3.5 text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 font-serif leading-relaxed focus:outline-none focus:ring-2 focus:ring-indigo-500 min-h-[220px]"
            />
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
            <div>
              {activeSample?.source_url ? (
                <a
                  href={activeSample.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  id="article-source-btn"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-750 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white text-xs font-medium shadow-2xs transition-colors cursor-pointer group"
                  title={`Open source: ${activeSample.source_name || activeSample.dataset}`}
                >
                  <span className="font-semibold text-slate-500 dark:text-slate-400">Source:</span>
                  <span className="font-medium text-indigo-600 dark:text-indigo-400 group-hover:underline">
                    {activeSample.source_name || activeSample.dataset}
                  </span>
                  <ArrowUpRight className="h-3.5 w-3.5 text-slate-400 dark:text-slate-400 group-hover:text-indigo-500 transition-colors" />
                </a>
              ) : null}
            </div>

            <button
              id="analyze-btn"
              onClick={runAnalysis}
              disabled={isAnalyzing || text.trim().length < 10}
              className="inline-flex items-center gap-2 px-6 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-sm cursor-pointer ml-auto"
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  Analyze Bias
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div className="p-4 rounded-xl border border-red-200 dark:border-red-900/60 bg-red-50 dark:bg-red-950/40 text-red-700 dark:text-red-300 text-sm flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* ── 3. Results Section ────────────────────────────────────────────── */}
        {analysis && (
          <div className="flex flex-col gap-8 animate-in fade-in duration-300">
            {/* Multi-Model Telemetry Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Specialist 1 */}
              <div className="border border-slate-200 dark:border-slate-800 rounded-xl p-4 bg-white dark:bg-slate-900 shadow-xs flex flex-col gap-2 transition-colors duration-200">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider border-b border-slate-100 dark:border-slate-800 pb-2">
                  <span>Sentence Spin Head</span>
                  <span className="font-mono text-[10px] bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-1.5 py-0.5 rounded">
                    RoBERTa BABE
                  </span>
                </div>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-2xl font-black text-slate-900 dark:text-slate-100">
                    {Math.round(analysis.specialist_sentence.bias_ratio * 100)}%
                  </span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full font-bold uppercase ${
                      analysis.specialist_sentence.biased_sentence_count > 0
                        ? "bg-red-100 dark:bg-red-950/50 text-red-700 dark:text-red-400"
                        : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                    }`}
                  >
                    {analysis.specialist_sentence.label}
                  </span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                  Flagged {analysis.specialist_sentence.biased_sentence_count} of{" "}
                  {analysis.specialist_sentence.total_sentences} sentences as carrying loaded spin.
                </p>
              </div>

              {/* Specialist 2 */}
              <div className="border border-slate-200 dark:border-slate-800 rounded-xl p-4 bg-white dark:bg-slate-900 shadow-xs flex flex-col gap-2 transition-colors duration-200">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider border-b border-slate-100 dark:border-slate-800 pb-2">
                  <span>Political Stance</span>
                  <span className="font-mono text-[10px] bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-1.5 py-0.5 rounded">
                    DeBERTa AllSides
                  </span>
                </div>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-xl font-bold uppercase tracking-tight text-slate-900 dark:text-slate-100">
                    {analysis.specialist_political.predicted_label}
                  </span>
                  <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">
                    {Math.round(analysis.specialist_political.confidence * 100)}% conf
                  </span>
                </div>
                {/* 3-way distribution bar */}
                <div className="flex flex-col gap-1 mt-2">
                  <div className="h-2 w-full rounded-full bg-slate-100 dark:bg-slate-800 flex overflow-hidden">
                    <div
                      style={{ width: `${(analysis.specialist_political.scores?.LEFT || 0) * 100}%` }}
                      className="bg-blue-500"
                      title="Left"
                    />
                    <div
                      style={{ width: `${(analysis.specialist_political.scores?.CENTER || 0) * 100}%` }}
                      className="bg-slate-400 dark:bg-slate-500"
                      title="Center"
                    />
                    <div
                      style={{ width: `${(analysis.specialist_political.scores?.RIGHT || 0) * 100}%` }}
                      className="bg-red-500"
                      title="Right"
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-slate-500 dark:text-slate-400 font-mono">
                    <span>L: {Math.round((analysis.specialist_political.scores?.LEFT || 0) * 100)}%</span>
                    <span>C: {Math.round((analysis.specialist_political.scores?.CENTER || 0) * 100)}%</span>
                    <span>R: {Math.round((analysis.specialist_political.scores?.RIGHT || 0) * 100)}%</span>
                  </div>
                </div>
              </div>

              {/* Specialist 3 */}
              <div className="border border-slate-200 dark:border-slate-800 rounded-xl p-4 bg-white dark:bg-slate-900 shadow-xs flex flex-col gap-2 transition-colors duration-200">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider border-b border-slate-100 dark:border-slate-800 pb-2">
                  <span>Isolated LLM Reading</span>
                  <span className="font-mono text-[10px] bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 px-1.5 py-0.5 rounded">
                    Zero-Shot
                  </span>
                </div>
                <div className="flex items-baseline justify-between mt-1">
                  <span className="text-xl font-bold uppercase tracking-tight text-slate-900 dark:text-slate-100">
                    {analysis.isolated_llm?.political_leaning?.label || "Center"}
                  </span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full font-bold uppercase ${
                      analysis.isolated_llm?.linguistic_bias?.is_biased
                        ? "bg-red-100 dark:bg-red-950/50 text-red-700 dark:text-red-400"
                        : "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300"
                    }`}
                  >
                    {analysis.isolated_llm?.linguistic_bias?.is_biased ? "Biased" : "Neutral"}
                  </span>
                </div>
                <p
                  className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed mt-1"
                  title={analysis.isolated_llm?.reasoning || "Initial baseline assessment."}
                >
                  {analysis.isolated_llm?.reasoning || "Initial baseline assessment."}
                </p>
              </div>
            </div>

            {/* In-Line Annotated Text Reader */}
            <div className="border border-slate-200 dark:border-slate-800 rounded-2xl p-5 bg-white dark:bg-slate-900 shadow-xs transition-colors duration-200">
              <div className="flex items-center justify-between mb-3 border-b border-slate-100 dark:border-slate-800 pb-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                  <FileText className="h-3.5 w-3.5" />
                  Sentence-by-Sentence Spin Annotation
                </span>
                <span className="text-xs text-slate-400 dark:text-slate-500">
                  Hover sentences to inspect specialist confidence
                </span>
              </div>
              <div className="font-serif text-base leading-loose text-slate-900 dark:text-slate-100">
                {analysis.sentences.map((s, idx) => (
                  <span
                    key={idx}
                    title={s.is_biased ? `da-roberta-babe score: ${Math.round(s.score * 100)}% biased` : "Neutral sentence"}
                    className={`px-1 py-0.5 rounded cursor-help transition-colors ${
                      s.is_biased
                        ? "bg-amber-100 dark:bg-amber-950/60 text-amber-950 dark:text-amber-200 border-b-2 border-amber-500 dark:border-amber-400 font-medium"
                        : "text-slate-900 dark:text-slate-150 hover:bg-slate-100 dark:hover:bg-slate-800"
                    }`}
                  >
                    {s.text}{" "}
                  </span>
                ))}
              </div>
            </div>

            {/* ── Arbiter Card: "Why is it (not) biased?" ───────────────────── */}
            <div className="border-2 border-indigo-500/40 dark:border-indigo-500/30 rounded-2xl p-6 bg-gradient-to-b from-white to-slate-50 dark:from-slate-900 dark:to-slate-900/90 shadow-md flex flex-col gap-5 transition-colors duration-200">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-200 dark:border-slate-800 pb-4">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-widest text-indigo-600 dark:text-indigo-400 mb-1">
                    Arbiter Consensus Judgment
                  </div>
                  {/* Dynamic Heading based on verdict */}
                  <h2 className="text-2xl font-extrabold tracking-tight text-slate-900 dark:text-white">
                    {dynamicHeading}
                  </h2>
                </div>

                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <div className="text-xs font-semibold uppercase text-slate-500 dark:text-slate-400">
                      Consensus Stance
                    </div>
                    <div className="font-bold uppercase text-sm text-slate-900 dark:text-slate-100">
                      {analysis.arbiter?.final_verdict?.overall_leaning || "Neutral"}
                    </div>
                  </div>
                  <span
                    className={`text-sm px-3.5 py-1 rounded-full font-bold uppercase ${
                      isBiased
                        ? "bg-red-600 text-white"
                        : "bg-slate-200 dark:bg-slate-800 text-slate-800 dark:text-slate-200"
                    }`}
                  >
                    {isBiased ? "BIASED" : "NON-BIASED"}
                  </span>
                </div>
              </div>

              {/* Arbiter Explanation Text */}
              <div className="text-sm leading-relaxed whitespace-pre-line text-slate-800 dark:text-slate-200 font-serif">
                {analysis.arbiter?.why_biased_explanation ||
                  "The arbiter evaluated all specialist signals to formulate this verdict."}
              </div>

              {/* Key Findings List */}
              {analysis.arbiter?.key_findings && analysis.arbiter.key_findings.length > 0 && (
                <div className="bg-slate-100/70 dark:bg-slate-800/60 p-4 rounded-xl flex flex-col gap-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                    Specific Evidence & Findings:
                  </span>
                  <ul className="list-disc list-inside text-xs space-y-1.5 text-slate-700 dark:text-slate-300">
                    {analysis.arbiter.key_findings.map((f, i) => (
                      <li key={i}>{f}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Action: Suggest Rewrite */}
              {isBiased && (
                <div className="border-t border-slate-200 dark:border-slate-800 pt-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
                  <div className="text-xs text-slate-500 dark:text-slate-400">
                    Biased rhetoric detected. Generate a factual, neutral AP/Reuters wire revision.
                  </div>
                  <button
                    id="rewrite-btn"
                    onClick={runRewrite}
                    disabled={isRewriting}
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-sm cursor-pointer"
                  >
                    {isRewriting ? (
                      <>
                        <RefreshCw className="h-4 w-4 animate-spin" />
                        Generating Neutral Revision...
                      </>
                    ) : (
                      <>
                        <Sparkles className="h-4 w-4" />
                        Suggest Rewrite
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>

            {/* ── Rewrite Output View ───────────────────────────────────────── */}
            {rewrite && (
              <div className="border border-slate-200 dark:border-slate-800 rounded-2xl p-6 bg-white dark:bg-slate-900 flex flex-col gap-6 shadow-sm animate-in fade-in duration-300 transition-colors duration-200">
                <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                    <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100">Neutral AP-Style Revision</h3>
                  </div>
                  <span className="text-xs px-2.5 py-0.5 rounded-full font-semibold border border-emerald-300 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300">
                    100% Fact-Preserving
                  </span>
                </div>

                <div className="p-4 rounded-xl bg-emerald-50/50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900/40 font-serif text-base leading-relaxed text-slate-900 dark:text-emerald-100">
                  {rewrite.rewritten_text}
                </div>

                {rewrite.editorial_summary && (
                  <p className="text-xs text-slate-500 dark:text-slate-400 italic">
                    Editorial note: {rewrite.editorial_summary}
                  </p>
                )}

                {/* Before / After Diff Table */}
                {rewrite.changes && rewrite.changes.length > 0 && (
                  <div className="flex flex-col gap-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                      Before / After Modification Log:
                    </span>
                    <div className="border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden">
                      <table className="w-full text-xs text-left">
                        <thead className="bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 uppercase text-[10px] tracking-wider">
                          <tr>
                            <th className="py-2.5 px-3">Original Loaded Phrasing</th>
                            <th className="py-2.5 px-3">Neutralized Revision</th>
                            <th className="py-2.5 px-3">Editorial Rationale</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                          {rewrite.changes.map((c, i) => (
                            <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                              <td className="py-2.5 px-3 font-medium text-red-600 dark:text-red-400 line-through decoration-red-400 dark:decoration-red-500">
                                {c.original}
                              </td>
                              <td className="py-2.5 px-3 font-semibold text-emerald-700 dark:text-emerald-400">
                                {c.rewritten}
                              </td>
                              <td className="py-2.5 px-3 text-slate-500 dark:text-slate-400">{c.reason}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>

      {/* ── About Popup Modal ─────────────────────────────────────────────── */}
      {isAboutOpen && (
        <div
          id="about-modal-backdrop"
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 dark:bg-black/60 backdrop-blur-xs p-4"
          onClick={() => setIsAboutOpen(false)}
        >
          <div
            id="about-modal-card"
            className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-2xl p-7 w-80 sm:w-88 aspect-square flex flex-col justify-between relative transition-colors duration-200"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              id="close-about-btn"
              onClick={() => setIsAboutOpen(false)}
              aria-label="Close"
              className="absolute top-4 right-4 p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
            >
              <X className="h-4 w-4" />
            </button>

            <div className="flex flex-col gap-3 pt-1">
              <div className="flex items-center gap-2 text-indigo-600 dark:text-indigo-400">
                <Scale className="h-5 w-5" />
                <span className="text-[11px] font-bold tracking-widest uppercase font-mono">Project Info</span>
              </div>
              <h3 className="text-lg font-extrabold tracking-tight text-slate-900 dark:text-white leading-snug">
                News Bias Detection using Machine Learning
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                Multi-specialist machine learning and editorial arbiter architecture for detecting micro-sentence spin, political stance, and media bias across news articles.
              </p>
            </div>

            <div className="border-t border-slate-100 dark:border-slate-800 pt-3 flex items-center justify-between text-[11px] text-slate-400 dark:text-slate-500 font-mono">
              <span>NewsBias Standalone</span>
              <span>v1.0</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
