# Contextual audit of final checkpoint explanations

This is a manual qualitative audit of all 100 examples in
[`final_checkpoint_examples.md`](final_checkpoint_examples.md). The binary verdict answers:
**Is the explanation's overall account plausible and supported by the available context?**

## Rubric and limitations

- **Correct** means the main functional description is contextually plausible and does not add a materially unrelated entity, domain, event, or causal story.
- **Incorrect** means at least one material part is contradictory, incoherent, or unsupported. A correct token-level observation does not rescue an invented semantic narrative.
- Generic but context-compatible explanations can pass, although the rationale marks when they are weak or low-information.
- The Markdown file displays only the final 700 characters of each prefix. The full saved prefixes in `rl_samples.json` were also inspected.
- As an auxiliary check, the frozen target readouts were recomputed with `Qwen/Qwen2.5-0.5B-Instruct` at pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`. All reconstructed prefixes matched their recorded token counts. This verifies some literal strengthened/suppressed-token claims, but the verdict remains a semantic/contextual judgment.
- This is one evaluator's binary annotation of a convenience sample (the first 100 of 5,000 eval examples), not a blinded or inter-annotator human evaluation.

## Results

**48 correct; 52 incorrect.** Format validity was 99/100, so syntactic validity substantially overstates semantic quality in this sample.

| # | Row | Verdict | Rationale |
|---:|:---|:---:|:---|
| 1 | `72:119` | Incorrect | Correctly notices movement away from “know,” but invents football teams in a golf article. |
| 2 | `72:156` | Incorrect | Action continuations are plausible, but “losing control over sports performance options” is unsupported. |
| 3 | `72:241` | Incorrect | The father-son record concerns golf; the explanation changes it to baseball history. |
| 4 | `91:458` | Correct | An Orange-Apple deal and Apple product positioning are the supported continuation; the Apple-token suppression is also real. |
| 5 | `91:614` | Incorrect | Firmware unlocking and hackers are in context, but a malware incident is invented. |
| 6 | `91:896` | Incorrect | Contrast or elaboration is plausible, but this is an iPhone article, not a show. |
| 7 | `115:482` | Incorrect | The “cry” suppression is real, but specific funeral-service times are not supported. |
| 8 | `115:779` | Correct | The polemic explicitly builds toward a spectacular anti-terror response, so an action/justice continuation is plausible. |
| 9 | `115:839` | Incorrect | The unfinished `3,` likely begins a commemorative number; hours after midnight are unsupported. |
| 10 | `228:87` | Correct | The text is entering a list of attempted Windows/system troubleshooting steps. |
| 11 | `228:179` | Incorrect | `03:` is an unfinished timestamp and calls for digits, not a shift away from numerical values. |
| 12 | `228:513` | Correct | Moving from the wired NIC noun phrase into an action/conjunction and further troubleshooting is supported. |
| 13 | `243:261` | Correct | The sentence can close after “Earth” and proceed to further space-radiation details. |
| 14 | `243:289` | Incorrect | `Approx` continues naturally to “Approximately”; the formatting-symbol account misses this lexical completion. |
| 15 | `243:599` | Correct | Spacecraft, space-radiation, protection, and possible medical applications all follow from the scientific context. |
| 16 | `252:121` | Incorrect | Although digits are relatively suppressed by the delta, the model still predicts a digit to finish `20`; “varied sentence endings” misreads that relative change. |
| 17 | `252:144` | Correct | The conjunction introduces another named official in an FCC/corporate meeting. |
| 18 | `252:149` | Incorrect | The China Securities Association and the alleged financial matter are wholly absent. |
| 19 | `253:80` | Incorrect | The phrase is headed toward pedestrian/bicycle facilities, paths, or routes, not vehicle types. |
| 20 | `253:86` | Correct | The next idea is how built-environment characteristics influence physical activity and health outcomes. |
| 21 | `253:146` | Correct | Sentence closure after “planning and zoning commissions” and a conclusion about local organizations are plausible. |
| 22 | `267:208` | Incorrect | This is an analyst attribution in a router-chip article, not a transition to financial markets. |
| 23 | `267:222` | Incorrect | “Technically” introduces router throughput/specifications, not device compatibility or software updates. |
| 24 | `267:339` | Correct | “Huawei recently began selling a …” strongly supports a new router/chip/product continuation. |
| 25 | `268:391` | Correct | It continues a university grant's prohibited-expense list; project costs and office-related expenses fit. |
| 26 | `268:848` | Correct | Assurances being obtained, reviewed, or approved before recruitment is the expected formal status. |
| 27 | `268:894` | Correct | “Seed” is leading into Seed Grant funds/equipment administration. |
| 28 | `288:102` | Incorrect | The text says newly engaged; childbirth after cancer is fabricated. |
| 29 | `288:116` | Incorrect | Will Kopelman is visible in the surrounding sequence; “Mike” and an already completed marriage are unsupported. |
| 30 | `288:123` | Correct | Leaving a hospital, clinic, doctor, or maternity-related location is plausible after the sonogram setup. |
| 31 | `336:142` | Correct | The boundary is inside a pizza-size enumeration, so another digit/measurement continuation is appropriate. |
| 32 | `336:193` | Incorrect | The actual discourse turns to whether there is enough demand; cheese, vegetables, and home kitchens are speculative. |
| 33 | `336:214` | Incorrect | A first-person auxiliary is likely, but past work with others at home is an invented scenario. |
| 34 | `383:411` | Correct | A description of the girlfriend and their relationship after his return to Australia is supported. |
| 35 | `383:598` | Incorrect | The passage concerns injury and commitment; it contains no abuse. |
| 36 | `383:669` | Incorrect | “Worst time in my life/career” is well identified, but college years are invented. |
| 37 | `394:186` | Correct | A newsletter introduction naturally proceeds to its resources, topics, or upcoming features, though this is generic. |
| 38 | `394:711` | Incorrect | The passage explains adding armies in a strategy-game calculator, not test-session data processing. |
| 39 | `394:836` | Incorrect | Winning/success percentages are right, but the domain is a strategy game rather than sports. |
| 40 | `442:69` | Correct | The boundary is in a weather forecast and clearly prepares temperature/precipitation detail. |
| 41 | `442:171` | Incorrect | The local syntax is captured, but snowfall is incompatible with the warm, muggy summer forecast. |
| 42 | `442:227` | Incorrect | Friday cooling is supported; winter climate is not. |
| 43 | `461:683` | Correct | A third-place cycling finish followed by race-result/team-performance detail is directly supported. |
| 44 | `461:802` | Correct | The text is elaborating on the cycling team's exertion and tactical effort, albeit the explanation is vague. |
| 45 | `461:862` | Correct | A day name or further event/race result naturally follows the dangling “On”. |
| 46 | `466:79` | Incorrect | The sentence continues describing a font's two widths; keyboard settings are unrelated. |
| 47 | `466:86` | Incorrect | The widths contain stylistic alternates, not colors or image-size features. |
| 48 | `466:108` | Correct | Further OpenType/design-tool behavior is a supported continuation. |
| 49 | `475:111` | Correct | “Rejoined the ranks of 1A athletics” is the exact kind of team-history continuation predicted. |
| 50 | `475:199` | Correct | The headline is leading into a game result and team-performance details. |
| 51 | `475:249` | Incorrect | The visible event is boys basketball, not football. |
| 52 | `533:76` | Correct | A numerical market level or related economic indicator naturally follows “started at close to”. |
| 53 | `533:253` | Correct | Participation by more credit unions and the program's reach/outcomes are supported. |
| 54 | `533:881` | Correct | The sentence is finishing or introducing figures about retirement savings targets; it is accurate but low-information. |
| 55 | `621:77` | Correct | The sentence enumerates payment/account-access channels, including online, ATM, and point-of-sale access. |
| 56 | `621:153` | Incorrect | `340,` continues to `000 ATMs`; “million” and a shift away from numbers are wrong. |
| 57 | `621:245` | Incorrect | “Last year/month” is plausible, but holiday promotions are invented. |
| 58 | `622:116` | Incorrect | An international congress/meeting is plausible, but UNESCO and the UN are unsupported named organizations. |
| 59 | `622:140` | Correct | Ongoing or previously implemented trinational wilderness programs follow naturally. |
| 60 | `622:178` | Incorrect | “First” is plausible, but the passage is North American; the UK and Europe are fabricated. |
| 61 | `681:78` | Incorrect | `pall` is a truncated “pallid,” not a cue for `-ing` words about beauty or power. |
| 62 | `681:143` | Incorrect | The point is the risk of banality/foolishness, not perfection. |
| 63 | `681:170` | Incorrect | The likely issue is the fourth estate's monopoly or privileged status, not school music/radio policy. |
| 64 | `799:117` | Correct | Strict melodic/rhythmic limits, frameworks, rules, or boundaries are exactly supported. |
| 65 | `799:155` | Incorrect | It is a critic's byline (David Sokol in the subsequent text), not “David Smith.” |
| 66 | `799:171` | Incorrect | The boundary likely closes a review quotation; `-of`/`day` and a detailed-work narrative add little supported information. |
| 67 | `842:185` | Incorrect | “Quick tournament/game” is plausible, but winning streaks contradict the discussion of tournament duration. |
| 68 | `842:481` | Incorrect | `2–4` is a player-count category; millions and user-engagement rankings are unrelated. |
| 69 | `842:629` | Correct | The passage proceeds to later tournament-running actions, so a sequential instruction continuation is plausible. |
| 70 | `890:73` | Incorrect | The continuation is “one of my least favorite restaurants,” not a list of additional meal choices. |
| 71 | `890:96` | Incorrect | `cr` begins “crabcake”; the letter/suffix account does not explain the semantic continuation. |
| 72 | `890:119` | Incorrect | Sentence-ending punctuation is plausible, but Friday nights across America are invented. |
| 73 | `908:78` | Incorrect | The discussion concerns a Street View image, a child, and a possible gun; New York traffic is unsupported. |
| 74 | `908:491` | Incorrect | The joking Martian/child/gun discussion has nothing to do with a racing collision. |
| 75 | `908:556` | Incorrect | The output degenerates into incoherent multilingual/code-like text and is format-invalid. |
| 76 | `1092:111` | Incorrect | Community beneficiaries are broadly plausible, but farmers, homeowners, and energy generation are unsupported specifics. |
| 77 | `1092:112` | Correct | Members/customers are plausible beneficiaries of member cooperatives; the token shift is also directly supported. |
| 78 | `1092:123` | Correct | Collective work leading to further cooperative/community action is contextually supported, though generic. |
| 79 | `1120:467` | Incorrect | The text concerns UK parties and Brexit; Ireland and an election setting are invented. |
| 80 | `1120:655` | Incorrect | `2002-2` continues a date, but financial figures and varied punctuation are unsupported. |
| 81 | `1120:807` | Correct | A Brexit deal, government decision, negotiations, or consequence is the supported continuation. |
| 82 | `1125:74` | Correct | The source is noisy spam, but a continuation about the referenced health product is context-compatible and cautious enough. |
| 83 | `1125:493` | Correct | Closing the analyzed-study clause and adding patient outcomes/treatment details are plausible. |
| 84 | `1125:568` | Correct | A medical-policy subject such as an individual/patient/member and treatment status fits the local context. |
| 85 | `1139:88` | Correct | Punctuation or qualification after “often do” and a nuanced dietary-choice continuation are supported. |
| 86 | `1139:165` | Correct | “Breakfast” is a section heading leading into recipes/options. |
| 87 | `1139:248` | Correct | The dangling `L` likely begins “Lunch,” so another meal-options section is supported despite weak token wording. |
| 88 | `1193:99` | Incorrect | Investments/initiatives are correctly identified, but Amazon and Google are fabricated. |
| 89 | `1193:173` | Incorrect | A facility/location continuation is right, but the guessed countries and “nearby facilities” are unsupported. |
| 90 | `1193:414` | Correct | Innovation, solutions, quality, and product benefits fit CP Kelco's corporate description. |
| 91 | `1234:64` | Correct | “More information/details” about the dental condition and patient case is directly supported. |
| 92 | `1234:65` | Correct | Additional image views and linked case content fit the page structure. |
| 93 | `1234:68` | Correct | Closing the case-details link and moving to medical articles/resources is plausible. |
| 94 | `1240:234` | Incorrect | The page is sequencing comics/dialogue; detailed responses or comments about the story are not established. |
| 95 | `1240:357` | Incorrect | The political joke does not concern being late at home. |
| 96 | `1240:478` | Incorrect | Digits are correctly expected after “Share June,” but they form a date, not a quiz score. |
| 97 | `1266:127` | Correct | Cash/in-person or another payment method naturally follows “fees can be paid in”. |
| 98 | `1266:336` | Correct | A numeric monetary amount follows the dollar sign, within the stated school-fee policy. |
| 99 | `1266:377` | Correct | “Additional fee” and related athletic-program financial terms are directly supported. |
| 100 | `1326:78` | Correct | Human responsibility, freedom, and divine authority are plausible themes in this religious argument. |

## When the delta NLA does well

The clearest successes occur when the next-token computation is locally constrained and the explanation stays near that evidence:

1. **Common syntactic or lexical slots.** Examples include an official after “and” (#17), a product after “selling a” (#24), a measurement (#31), “ranks of” (#49), payment channels (#55), and “strict … frameworks” (#64).
2. **Structured lists and templated pages.** The grant sequence (#25–27), cycling report (#43–45), meal plan (#85–87), medical case page (#91–93), and athletic-fee policy (#97–99) are consistently good.
3. **Strong, repeated domain cues.** Weather (#40), finance/payment systems (#52–55), medical policy (#83–84), and theology (#100) give the delta a narrow semantic basin.
4. **Cautious abstraction.** Explanations that say “continues the grant-expense list” or “prepares further race results” fare better than explanations that guess an unseen person, country, sport, or event.

## When it underperforms

1. **Unsupported semantic completion after a real token shift.** This is the dominant failure. For example, the model really suppresses `know` in #1 but then invents football; it captures `will → be` in #41 but invents snowfall; it captures investment-related movement in #88 but invents Amazon and Google; and it predicts digits in #96 but calls them a quiz score instead of a date.
2. **Proper nouns, geography, and specialized domains.** Golf becomes football/baseball (#1, #3), an FCC filing becomes the China Securities Association (#18), Britain becomes Ireland (#79), and a corporate facility list acquires guessed countries (#89).
3. **Partial tokens, dates, and numbers.** In the manually identified hard-boundary subset (#9, #11, #14, #16, #31, #56, #61, #68, #71, #80, #87, #96, #98), only #31, #87, and #98 pass: 3/13. The NLA often describes raw token-shape movement without recovering the composed word, date, or quantity.
4. **Boilerplate overreach.** Many outputs have a reasonable first clause, then append a stock narrative such as “at various locations,” “after winning streaks,” or “within schools.” Those additions turn otherwise partial successes into failures.
5. **Context-free verbalization.** The AV receives the delta but not the original text. It often recovers broad grammatical or predictive structure from the delta while lacking enough referential information to attach that structure to the right person, sport, location, or event.
6. **The RL objective checks reconstructability, not truth.** The AR can reconstruct useful delta information from a fluent but semantically false phrase. Format penalties explain the 99% validity rate; they do not prevent hallucinated meaning.

## Bottom line

This checkpoint is already a useful **decoder of broad endpoint effects**: syntax, continuation type, coarse topic, and some salient candidate changes. It is not yet a reliable natural-language explanation system. Its characteristic error is not random gibberish—only one example does that—but a diagnostically informed, fluent statement followed by an unsupported story. A future evaluation should therefore report semantic groundedness separately from FVE, prediction KL, and format validity.
