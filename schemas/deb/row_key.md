# DEB row alignment key

Predicted rows are matched to gold rows with this key (after normalization).

## Primary key

`(person_key, address_type, address_source)`

1. **person_key** — normalize `person_name` (`Name (Borrwer,Co-applicant,Gurantor)`); if empty, fall back to `customer_name` (`CUSTOMER_NAME`).
2. **address_type** — `address_type` / Type Of Address (`H1`, `H2`, `W`, …), uppercased, whitespace stripped.
3. **address_source** — `address_source` canonicalized via known aliases (see below).

## Disambiguator

When the primary key collides within one account, also match on normalized `address_line1`.

## Unmatched rows

- Gold row with no prediction → **false negative** (hurts row recall).
- Prediction with no gold → **false positive** (hurts row precision).
- Field-level metrics run **only on aligned pairs**.

## Address-source aliases (canonical → forms)

| Canonical | Accepted forms |
|-----------|----------------|
| `applicationform` | Application form, Application Form, ApplicationForm, Applicationform |
| `voterid` | Voterid, Voter id, Voter ID, VoterId |
| `rationcard` | Rationcard, Ration card, Ration Card |
| `aadhaarcard` | Aadhaarcard, Aadhaar card, Aadhaar |
| `dl` | DL, Driving Licence, Driving License |
| `agreement_of_guarantee` | Agreement of Guarantee, Aggreement of Guarantee, Aggrement of Guarantee, Agreement Of Gurantee |
| `dpn` | DPN |
| `guarantor_form` | Guarantor's form, Guarantor form |

Unknown sources: casefold + collapse whitespace + strip non-alnum separators to underscores.
