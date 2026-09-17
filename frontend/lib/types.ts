export interface DatasetSummary {
  dataset_id: string;
  source_name: string;
  source_type: string;
  row_count: number;
  column_count: number;
  schema: Record<string, string>;
  ingested_at: string;
  profiled: boolean;
}

export interface ColumnProfile {
  name: string;
  physical_type: string;
  data_type: string;
  semantic_type: string;
  semantic_confidence: number;
  semantic_evidence: Record<string, number>;
  null_count: number;
  null_pct: number;
  unique_count: number;
  uniqueness: number;
  cardinality: string;
  min: unknown;
  max: unknown;
  mean: number | null;
  std: number | null;
  sample_values: unknown[];
  top_values: [unknown, number][];
  pattern_histogram: Record<string, number>;
  date_format: string | null;
}

export interface DatasetProfile {
  dataset_id: string;
  row_count: number;
  column_count: number;
  duplicate_rows: number;
  columns: ColumnProfile[];
  elapsed_ms: number;
}

export interface SessionInfo {
  session_id: string;
  datasets: DatasetSummary[];
  discovered: boolean;
  preferences: Preferences;
  plan_id: string | null;
  merges: { merge_id: string; status: string; rows: number; conflicts: number; created_at: string }[];
  chat_turns: number;
}

export interface Preferences {
  mode: string;
  conflict_strategy: string;
  confidence_threshold: number | null;
  primary_key: string | null;
  merge_uncertain: boolean | null;
  source_priority: string[] | null;
  thresholds: { auto_merge: number; review: number; min_edge: number };
  scorer?: string;
}

export interface ToolCallInfo {
  tool: string;
  arguments: Record<string, unknown>;
  ok: boolean;
  error?: string | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  tool_calls?: ToolCallInfo[];
  provider?: { provider: string; model: string; mock: boolean; fallback?: boolean };
  elapsed_ms?: number;
  pending?: boolean;
}

export interface ChatResponse {
  session_id: string;
  reply: string;
  tool_calls: ToolCallInfo[];
  cards: { type: string; tool: string; data: unknown }[];
  suggestions: string[];
  provider: { provider: string; model: string; mock: boolean; fallback?: boolean };
  elapsed_ms: number;
}

export interface Factor {
  signal: string;
  label: string;
  value: number;
  weight: number;
  contribution: number;
}

export interface MatchExplanation {
  left: string;
  right: string;
  confidence: number;
  band: string;
  accepted: boolean;
  relationship: string;
  rejection_reason: string | null;
  factors: Factor[];
  strongest_factor: string;
  weakest_factor: string;
  normalizer: string;
  graph: { base_score: number; adjustment: number; transitive_support: number; structural_support: number; exclusivity: number };
  learned_probability?: number | null;
  text: string;
}

export interface FlowView {
  level: "dataset" | "column";
  nodes: { id: string; type: string; position: { x: number; y: number }; data: Record<string, any>; parentId?: string; extent?: string; style?: Record<string, any> }[];
  edges: { id: string; source: string; target: string; label: string; data: Record<string, any> }[];
  stats: { nodes: number; edges: number; node_kinds: Record<string, number>; edge_types: Record<string, number>; cost_mode: string };
}

export interface MergeStep {
  step: number;
  operation: string;
  description: string;
  left: string | null;
  right: string | null;
  join_kind: string;
  join_type: string | null;
  join_keys: Record<string, string>[];
  confidence: number | null;
  threshold: number | null;
  requires_review: boolean;
}

export interface MergePlan {
  plan_id: string;
  mode: string;
  conflict_strategy: string;
  thresholds: { auto_merge: number; review: number; min_edge: number };
  root_dataset: string;
  grain: string;
  datasets: string[];
  steps: MergeStep[];
  canonical_schema: { name: string; semantic_type: string; data_type: string; sources: { dataset_id: string; column: string }[]; role: string | null; transformation: string | null }[];
  excluded_relationships: { left: string; right: string; confidence: number; reason: string; join_kind?: string }[];
  integration_tree: { parent: string; child: string; join_kind: string; confidence: number; key_pairs: Record<string, any>[]; requires_review: boolean; explanation: string }[];
  entity_groups: { group_id: string; datasets: string[]; anchor: string; method: string; fields: Record<string, any>[] }[];
  unmerged_datasets: string[];
  routes: { target: string; path: string[]; reliability: number; direct_confidence: number | null }[];
  warnings: string[];
}

export interface Conflict {
  entity_id: string;
  attribute: string;
  candidates: { value: unknown; source_dataset: string; source_column: string; source_row: number; confidence: number; record_timestamp: string | null; selected: boolean }[];
  resolved_value: unknown;
  strategy: string;
  reason: string;
  status: string;
}

export interface ReviewRecord {
  dataset: string;
  row: number;
  source: string;
  values: Record<string, unknown>;
}

export interface ReviewPair {
  left: ReviewRecord;
  right: ReviewRecord;
  probability: number;
  same_entity_now: boolean;
  score: number;
  sampling: "uncertainty" | "random";
  group_id: string;
  field_levels: Record<string, string>;
}

export interface ReviewQueue {
  merge_id: string;
  labels: number;
  calibration_labels: number;
  pairs: ReviewPair[];
  note: string;
}

export interface HistoryRow {
  entity_id: string;
  attribute: string;
  value: unknown;
  valid_from: string | null;
  valid_to: string | null;
  is_current: boolean;
  timestamp_kind: "change" | "creation" | "undated";
  supporting_records: number;
  source_dataset: string;
  source_column: string;
  source_row: number;
}

export interface HistoryResponse {
  merge_id: string;
  total: number;
  attributes_with_changes?: number;
  as_of?: string | null;
  rows: HistoryRow[];
  note?: string;
}

export interface SchemaColumn {
  name: string;
  description: string | null;
  semantic_type: string;
  data_type: string;
  role: "key" | "reference" | "linked" | "attribute";
  is_key: boolean;
  uniqueness: number;
  null_pct: number;
  connected: boolean;
  join_key: boolean;
}

export interface SchemaTable {
  id: string;
  label: string;
  source_type: string;
  rows: number | null;
  column_count: number;
  connected_columns: number;
  columns: SchemaColumn[];
  layer: number;
  order: number;
  component: number;
  unconnected: boolean;
}

export type LinkKind = "join_key" | "attribute" | "candidate" | "rejected";

export interface SchemaLink {
  id: string;
  from: { dataset: string; column: string };
  to: { dataset: string; column: string };
  kind: LinkKind;
  join_kind: string | null;
  relationship: string | null;
  confidence: number;
  band: string;
  status: string;
  in_merge_tree: boolean;
  normalizer: string;
  evidence: Record<string, number>;
  row_agreement?: number | null;
  rows_compared?: number | null;
  rejection_reason?: string | null;
}

export interface SchemaRelationship {
  id: string;
  from: string;
  to: string;
  join_kind: string;
  cardinality: string;
  confidence: number;
  band: string;
  in_merge_tree: boolean;
  key_pairs: { from: string; to: string }[];
  attribute_pairs: number;
  explanation: string;
}

export interface SchemaMap {
  level: "schema";
  tables: SchemaTable[];
  links: SchemaLink[];
  relationships: SchemaRelationship[];
  stats: { tables: number; relationships: number; links_by_kind: Record<string, number>; components: number; unconnected: string[] };
}
