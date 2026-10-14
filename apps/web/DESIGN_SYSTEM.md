# ModelXray · Ink / Olive

Current update: the UI uses only the existing light Stone / Olive palette. Theme switching and dark/system initialization have been removed. The original landing SVG diagram and four stage controls are restored. Selected rings pulse and incoming paths animate briefly when a stage is selected, with three repetitions under five seconds and reduced-motion support. The complete diagram scales within desktop panels; narrow screens retain local scrolling. Analytical workspace diagrams and all backend workflows remain unchanged. The original dark palette below is retained as design history, not an available theme.

Neutral surfaces carry the scientific workspace; olive identifies actions and selection. Information, failures and warnings use separate semantic colors. Charts contain returned measurements, never decorative analytics.

## Foundation

`src/app/tokens.css` owns both palettes, font aliases, spacing and motion. `globals.css` is the single component and responsive stylesheet. The previous override stylesheet has been removed. Compatibility aliases `accent2` and `accent3` resolve to olive; chart information uses `info`.

| Role | Dark | Light |
| --- | --- | --- |
| Canvas | `#0D0E0C` | `#F5F4EF` |
| Surface 1 | `#151813` | `#FFFFFF` |
| Surface 2 | `#191D17` | `#ECEDE6` |
| Surface 3 | `#232820` | `#E2E6DA` |
| Separator | `#32372C` | `#D6DACF` |
| Text | `#EEEFE9` | `#1B2119` |
| Secondary | `#B5BAAE` | `#515C4E` |
| Muted | `#919A88` | `#65715F` |
| Olive | `#A8B78B` | `#526B3F` |
| Olive hover | `#B9C9A0` | `#415532` |
| Selected surface | `#242D20` | `#E4EBDD` |
| Failure | `#D78C84` | `#9C463E` |
| Warning | `#D7B57B` | `#806020` |
| Information | `#94ADBE` | `#406880` |

Controls have stronger borders than noninteractive separators. Muted text is used on the canvas and primary surfaces; labels on denser secondary surfaces use secondary text for contrast. Success uses olive only where the underlying state supports success. An unloaded diagram is neutral. Its selected protocol step describes navigation, not job completion.

## Typography and spacing

Geist Variable is the primary font for headings, navigation, buttons and prose. IBM Plex Mono is reserved for feature conditions, identifiers, code and chart annotations. Both are self-hosted WOFF2 assets supplied by pinned Fontsource development dependencies. No third-party font requests or runtime font library are required. `next/font/local` handles loading and fallback. The sans font is preloaded; the technical font is not.

Body: 14px with 1.55 line height. Prose: 14–16px. Metadata: 12–13px. Workspace headings: 30–36px. Section headings: 20–26px. Landing display: 44–68px. Numbers use tabular numerals. Conditions wrap without truncating accessible evidence.

Spacing follows 4, 8, 12, 16, 24 and 32px increments. Panels and primary controls use 8px radii; inputs and navigation use 6px. Panels use quiet separators without glow or drop shadows. Metrics form a continuous neutral rail.

## Shell and states

The sticky header contains the wordmark, five workspaces, persistent theme control and primary action. A secondary row shows the current investigation identifier, actual API health and demo comparison. Evidence Ledger remains a collapsible utility. Navigation retains data and moves keyboard focus to the workspace.

Primary actions use olive with contrasting text; secondary controls use neutral surfaces. Active navigation exposes `aria-current`; protocol and filter selection expose `aria-pressed`. Disabled controls remain labeled. Uploads retain native file selection and filename feedback. Backend validation details remain visible; connection errors provide a readable explanation and retry actions.

Focus uses a 3px olive outline, including focus within file surfaces. Controls have at least 44px height. Errors use alerts; job states and filter counts use live regions. A skip link leads to workspace content. Progress comes only from the server, cancellation is cooperative, and terminal states use the returned job status.

## Analytical views

Overview combines actual totals, workflow relationships, effects and region links. Atlas retains master/detail inspection and source experiments. Experiments has labeled filters, an effect plot and a seven-column table with a sticky header. Coverage shows execution against generated candidates and feature blind spots. Regression retains model baselines and backend transitions.

SVG edges and node centers share one coordinate table. Paths describe relationships, never measured throughput. Protocol selection scrolls the stage into the local viewport. Plot and table regions accept keyboard focus. Mineral blue denotes confidence intervals, olive marks validated effects, and outlined points distinguish screened observations. Text and HTML tables provide color-independent alternatives. Charts explain units and basis. Discovery effects and holdout gaps remain separate. Regression plots use paired V2–V1 gaps without fabricated intervals.

Long plots initially show 12 entries while retaining the full axis domain and exact-value table. Expansion reveals all returned entries. Region actions lead to Atlas; source actions filter the actual experiment ID. Coverage remains bounded by generated hypotheses.

## Responsive behavior and motion

Content is capped at 1480px. Below 1180px navigation moves to a second row and the landing stacks. Below 900px analytical layouts and forms stack, metrics use two columns, and overflow stays inside local scroll regions. Below 620px controls wrap while preserving text and touch targets. Verification widths: 390, 768, 1024, 1280 and 1440px.

Theme initializes before paint, persists locally and falls back to system preference when storage is unavailable. Foreground and background switch immediately to prevent low-contrast intermediate theme colors. Motion is limited to border feedback. Reduced motion disables all transitions and animations.

Automated accessibility scans support review; manual assistive-technology and cross-browser certification remain separate work. See `VERIFICATION.md` for executed checks and limitations.
