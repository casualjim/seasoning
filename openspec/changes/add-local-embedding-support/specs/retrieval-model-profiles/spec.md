## ADDED Requirements

### Requirement: Embedding inputs model retrieval role
The system SHALL represent whether an embedding input is a query or a document.

#### Scenario: Query role is explicit
- **WHEN** a caller submits an embedding input with the query role
- **THEN** the system SHALL treat the input as retrieval query text during formatting

#### Scenario: Document role is explicit
- **WHEN** a caller submits an embedding input with the document role
- **THEN** the system SHALL treat the input as indexed document text during formatting

### Requirement: Query-side embedding instructions
The system SHALL support an embedding instruction or task description for retrieval queries, and SHALL apply it only to query formatting.

#### Scenario: Query instruction is applied to queries
- **WHEN** a caller supplies an embedding instruction and the input role is query
- **THEN** the system SHALL include that instruction in the formatted query text

#### Scenario: Query instruction is ignored for documents
- **WHEN** a caller supplies an embedding instruction and the input role is document
- **THEN** the system SHALL NOT inject that instruction into the formatted document text

### Requirement: Gemma retrieval formatting
For `ModelFamily::Gemma`, the system SHALL format embedding inputs according to EmbeddingGemma retrieval conventions.

#### Scenario: Gemma query formatting
- **WHEN** a caller embeds a query with `ModelFamily::Gemma`
- **THEN** the formatted query text SHALL be `task: <instruction-or-default> | query: <text>`

#### Scenario: Gemma document formatting with title
- **WHEN** a caller embeds a document with `ModelFamily::Gemma` and supplies a title
- **THEN** the formatted document text SHALL be `title: <title> | text: <text>`

#### Scenario: Gemma document formatting without title
- **WHEN** a caller embeds a document with `ModelFamily::Gemma` and does not supply a title
- **THEN** the formatted document text SHALL be `title: none | text: <text>`

### Requirement: Qwen3 retrieval formatting
For `ModelFamily::Qwen3`, the system SHALL format embedding inputs according to Qwen3 retrieval conventions.

#### Scenario: Qwen3 query formatting
- **WHEN** a caller embeds a query with `ModelFamily::Qwen3`
- **THEN** the formatted query text SHALL be `Instruct: <instruction-or-default>\nQuery: <text>`

#### Scenario: Qwen3 document formatting without title
- **WHEN** a caller embeds a document with `ModelFamily::Qwen3` and does not supply a title
- **THEN** the formatted document text SHALL be the raw document text

#### Scenario: Qwen3 document formatting with title
- **WHEN** a caller embeds a document with `ModelFamily::Qwen3` and supplies a title
- **THEN** the formatted document text SHALL contain the title, a newline, and the document text in that order

### Requirement: Formatting is crate-owned behavior
The crate SHALL apply family- and role-aware retrieval formatting as part of provider/client execution instead of requiring callers to pre-format retrieval strings.

#### Scenario: Caller provides semantic input only
- **WHEN** a caller supplies model family, embedding role, text, optional title, and optional instruction
- **THEN** the provider/client SHALL derive the formatted embedding text internally before transport or local inference
