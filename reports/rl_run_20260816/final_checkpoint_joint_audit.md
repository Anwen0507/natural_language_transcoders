# Joint semantic and token-direction audit

This audit applies an **AND rule** to all 100 examples in
[`final_checkpoint_examples.md`](final_checkpoint_examples.md):

> An explanation is correct only if (1) its semantic account is plausible or
> supported by the context, and (2) its stated suppressed and encouraged token
> information is directionally correct.

The semantic judgments are inherited from
[`final_checkpoint_explanation_audit.md`](final_checkpoint_explanation_audit.md).
For the token-direction judgment, the frozen target readouts were recomputed
with `Qwen/Qwen2.5-0.5B-Instruct` at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. A strict token pass requires the
claimed suppressed surface token or family to agree with a salient logit
decrease and every explicitly named encouraged example to have a positive
output-minus-input logit change. Obvious leading-space, BPE-fragment, and
inflection variants are treated as the same surface token.

## Result

| Semantic account | Token information | Count | Combined verdict |
|:---:|:---:|---:|:---:|
| Correct | Correct | **20** | Correct |
| Correct | Incorrect | 28 | Incorrect |
| Incorrect | Correct | 28 | Incorrect |
| Incorrect | Incorrect | 24 | Incorrect |

Thus **20/100 explanations are jointly correct** and **80/100 are incorrect**
under the requested definition.

## Per-example classification

| # | Row | Context | Tokens | Final | Concise rationale |
|---:|:---|:---:|:---:|:---:|:---|
| 1 | `72:119` | ✗ | ✓ | **Incorrect** | The `know` decrease and named increases are real, but football is invented in a golf article. |
| 2 | `72:156` | ✗ | ✗ | **Incorrect** | Neither the stated token shift nor the “losing control” sports narrative is supported. |
| 3 | `72:241` | ✗ | ✓ | **Incorrect** | The `Jay`/candidate directions are correct, but the father-son record is golf, not baseball. |
| 4 | `91:458` | ✓ | ✓ | **Correct** | The Apple suppression/candidate directions and the Orange-Apple deal interpretation both fit. |
| 5 | `91:614` | ✗ | ✓ | **Incorrect** | The named token directions hold, but a malware incident is invented. |
| 6 | `91:896` | ✗ | ✓ | **Incorrect** | Punctuation is suppressed and the named words rise, but this is not a show. |
| 7 | `115:482` | ✗ | ✓ | **Incorrect** | `cry` falls and the named continuations rise, but funeral-service times are unsupported. |
| 8 | `115:779` | ✓ | ✓ | **Correct** | The `show`-family suppression, named increases, and anti-terror action interpretation align. |
| 9 | `115:839` | ✗ | ✗ | **Incorrect** | The numerical/punctuation account and the midnight interpretation are both wrong. |
| 10 | `228:87` | ✓ | ✓ | **Correct** | The Windows surface-token shift and a transition into troubleshooting steps both fit. |
| 11 | `228:179` | ✗ | ✗ | **Incorrect** | The boundary is an unfinished timestamp; both the token account and narrative miss it. |
| 12 | `228:513` | ✓ | ✗ | **Incorrect** | Further NIC troubleshooting is plausible, but `wire` is not the salient suppressed token. |
| 13 | `243:261` | ✓ | ✓ | **Correct** | `Earth` falls, punctuation/prepositions rise, and further space-radiation detail follows. |
| 14 | `243:289` | ✗ | ✓ | **Incorrect** | The named directions have the claimed signs, but the semantic completion is “Approximately.” |
| 15 | `243:599` | ✓ | ✗ | **Incorrect** | The scientific interpretation fits, but the claimed `for` suppression does not. |
| 16 | `252:121` | ✗ | ✗ | **Incorrect** | It misinterprets relative digit suppression as punctuation/ending preparation inside a date. |
| 17 | `252:144` | ✓ | ✗ | **Incorrect** | Another named official is plausible, but `by` is not the token the delta saliently suppresses. |
| 18 | `252:149` | ✗ | ✗ | **Incorrect** | The token claims miss and the China Securities narrative is fabricated. |
| 19 | `253:80` | ✗ | ✗ | **Incorrect** | It misses the bicycle/facilities token transition and invents vehicle types. |
| 20 | `253:86` | ✓ | ✗ | **Incorrect** | The health/urban-planning interpretation fits, but `considering` is not the suppressed token. |
| 21 | `253:146` | ✓ | ✗ | **Incorrect** | Sentence closure is plausible, but “committees” does not match the suppressed `commissions` family. |
| 22 | `267:208` | ✗ | ✓ | **Incorrect** | The analyst/token directions are captured, but the article concerns router chips, not finance. |
| 23 | `267:222` | ✗ | ✗ | **Incorrect** | It misses the `Technically` fragment behavior and guesses software compatibility. |
| 24 | `267:339` | ✓ | ✓ | **Correct** | Article suppression and increased product/chip terms match Huawei selling a new router product. |
| 25 | `268:391` | ✓ | ✗ | **Incorrect** | The grant-expense interpretation is right, but the claimed `and` suppression is not. |
| 26 | `268:848` | ✓ | ✓ | **Correct** | The `be` family falls, formal-status candidates rise, and assurances are being obtained. |
| 27 | `268:894` | ✓ | ✗ | **Incorrect** | Seed Grant funding is correctly inferred, but `Award` is not the suppressed token. |
| 28 | `288:102` | ✗ | ✓ | **Incorrect** | The named token signs hold, but “newly” means newly engaged, not newborn after cancer. |
| 29 | `288:116` | ✗ | ✗ | **Incorrect** | The token claim misses and “Mike”/completed marriage contradicts the Will Kopelman context. |
| 30 | `288:123` | ✓ | ✓ | **Correct** | Article suppression and increased medical-location terms support a hospital/clinic continuation. |
| 31 | `336:142` | ✓ | ✓ | **Correct** | The numerical suppression and `inch` increase both fit the pizza-size enumeration. |
| 32 | `336:193` | ✗ | ✗ | **Incorrect** | It misses `enough`/demand and substitutes ingredients and home kitchens. |
| 33 | `336:214` | ✗ | ✓ | **Incorrect** | `I` falls and first-person auxiliaries rise, but past work at home is invented. |
| 34 | `383:411` | ✓ | ✗ | **Incorrect** | The relationship continuation is plausible, but `said` is not the suppressed token. |
| 35 | `383:598` | ✗ | ✓ | **Incorrect** | Punctuation falls and `But`/`She` rise, but there is no abuse. |
| 36 | `383:669` | ✗ | ✓ | **Incorrect** | `my → life/career/experience` is correct, but college years are fabricated. |
| 37 | `394:186` | ✓ | ✗ | **Incorrect** | Newsletter topics/resources are plausible, but `provide` is not the suppressed token. |
| 38 | `394:711` | ✗ | ✗ | **Incorrect** | Both the token account and the test-session narrative miss the strategy-game calculator. |
| 39 | `394:836` | ✗ | ✓ | **Incorrect** | The `of` decrease and winning terms are right, but this is a strategy game, not sports. |
| 40 | `442:69` | ✓ | ✗ | **Incorrect** | Weather/temperature is right, but not every named encouraged token has the claimed sign. |
| 41 | `442:171` | ✗ | ✓ | **Incorrect** | `will` falls and `be/get/continue` rise, but snowfall contradicts the hot forecast. |
| 42 | `442:227` | ✗ | ✗ | **Incorrect** | It misses the token shift and turns modest summer cooling into winter climate. |
| 43 | `461:683` | ✓ | ✗ | **Incorrect** | Race results are correctly inferred, but `berth` does not match the suppressed `place` family. |
| 44 | `461:802` | ✓ | ✓ | **Correct** | The past-auxiliary shift and increased effort descriptors fit the cycling-team narrative. |
| 45 | `461:862` | ✓ | ✗ | **Incorrect** | Another race/day is plausible, but the claimed commas and periods actually decrease. |
| 46 | `466:79` | ✗ | ✓ | **Incorrect** | `widths` falls and punctuation rises, but keyboard settings are unrelated to the font. |
| 47 | `466:86` | ✗ | ✓ | **Incorrect** | All named token signs hold, but colors/image sizes contradict the stylistic-alternates context. |
| 48 | `466:108` | ✓ | ✗ | **Incorrect** | The OpenType interpretation fits, but one named punctuation candidate decreases. |
| 49 | `475:111` | ✓ | ✓ | **Correct** | `ranks` falls, `of` rises, and “ranks of 1A athletics” is supported. |
| 50 | `475:199` | ✓ | ✗ | **Incorrect** | A sports result is correctly inferred, but `victorious` is not the suppressed token. |
| 51 | `475:249` | ✗ | ✓ | **Incorrect** | The named directions—including increased `football`—hold, but the visible sport is basketball. |
| 52 | `533:76` | ✓ | ✗ | **Incorrect** | A market-level continuation fits, but `$`, hyphen, comma, and period decrease rather than rise. |
| 53 | `533:253` | ✓ | ✗ | **Incorrect** | Program reach/outcomes fit, but `companies` does not match the suppressed `unions` family. |
| 54 | `533:881` | ✓ | ✗ | **Incorrect** | Retirement-savings figures fit, but `ideas` does not match the suppressed `targets` family. |
| 55 | `621:77` | ✓ | ✗ | **Incorrect** | Payment channels are right, but the explicitly named `all` token decreases. |
| 56 | `621:153` | ✗ | ✗ | **Incorrect** | `340,` continues to `000`; neither the suppression nor “million” account is correct. |
| 57 | `621:245` | ✗ | ✓ | **Incorrect** | `last` falls and time words/punctuation rise, but holiday promotions are invented. |
| 58 | `622:116` | ✗ | ✓ | **Incorrect** | The named token signs hold, but UNESCO and the UN are unsupported. |
| 59 | `622:140` | ✓ | ✓ | **Correct** | `have` falls, ongoing-program terms rise, and prior trinational cooperation is supported. |
| 60 | `622:178` | ✗ | ✓ | **Incorrect** | The token directions hold, but UK/European geography contradicts the North American context. |
| 61 | `681:78` | ✗ | ✓ | **Incorrect** | `pall` falls and `ing` rises, but the actual lexical completion is “pallid.” |
| 62 | `681:143` | ✗ | ✗ | **Incorrect** | It misses the `banality` fragment and invents a perfection contrast. |
| 63 | `681:170` | ✗ | ✓ | **Incorrect** | The named signs hold, but radio/music policy is unrelated to the fourth-estate argument. |
| 64 | `799:117` | ✓ | ✗ | **Incorrect** | Musical constraints are correctly inferred, but `rules` itself slightly decreases. |
| 65 | `799:155` | ✗ | ✓ | **Incorrect** | `David` falls and `Smith` rises slightly, but the actual byline is David Sokol. |
| 66 | `799:171` | ✗ | ✓ | **Incorrect** | The miracle/`of`/`day` directions hold, but the claimed artistic-work continuation is unsupported. |
| 67 | `842:185` | ✗ | ✓ | **Incorrect** | The speed/game token directions hold, but winning streaks misdescribe tournament duration. |
| 68 | `842:481` | ✗ | ✓ | **Incorrect** | The digit falls and `million` rises, but `2–4` denotes players, not millions. |
| 69 | `842:629` | ✓ | ✓ | **Correct** | `later` falls, sequential-action words rise, and further tournament instructions follow. |
| 70 | `890:73` | ✗ | ✗ | **Incorrect** | Although `one` falls, several claimed encouraged prepositions/punctuation marks decrease, and the semantic continuation is wrong. |
| 71 | `890:96` | ✗ | ✗ | **Incorrect** | It misses the `crabcake` fragment both lexically and semantically. |
| 72 | `890:119` | ✗ | ✓ | **Incorrect** | `week` falls and punctuation rises, but Friday nights across America are fabricated. |
| 73 | `908:78` | ✗ | ✓ | **Incorrect** | The named signs hold, but New York traffic is unsupported by the Street View discussion. |
| 74 | `908:491` | ✗ | ✗ | **Incorrect** | `walks` is not suppressed, `away` is decreased rather than encouraged, and the racing story is invented. |
| 75 | `908:556` | ✗ | ✗ | **Incorrect** | The output is incoherent/format-invalid and does not provide a reliable token account. |
| 76 | `1092:111` | ✗ | ✓ | **Incorrect** | The token directions hold, but farmers/homeowners and energy generation are unsupported specifics. |
| 77 | `1092:112` | ✓ | ✓ | **Correct** | `their` falls, member/customer terms rise, and cooperative beneficiaries are supported. |
| 78 | `1092:123` | ✓ | ✓ | **Correct** | `Together` falls, transition tokens rise, and further cooperative action is plausible. |
| 79 | `1120:467` | ✗ | ✗ | **Incorrect** | `Together` is not the suppressed token and Ireland/elections misstate the UK Brexit context. |
| 80 | `1120:655` | ✗ | ✗ | **Incorrect** | Digits—including claimed `0`—and punctuation decrease; the financial story also misses a date. |
| 81 | `1120:807` | ✓ | ✓ | **Correct** | Article suppression, Brexit-outcome increases, and the UK negotiation interpretation align. |
| 82 | `1125:74` | ✓ | ✗ | **Incorrect** | A health-product continuation is plausible, but `bes` is not the suppressed token. |
| 83 | `1125:493` | ✓ | ✓ | **Correct** | `analyzed` falls, transition/punctuation tokens rise, and study-outcome detail is plausible. |
| 84 | `1125:568` | ✓ | ✗ | **Incorrect** | The medical-policy interpretation fits, but the claimed punctuation candidates decrease. |
| 85 | `1139:88` | ✓ | ✗ | **Incorrect** | Dietary qualification is plausible, but the named hyphen/comma directions are wrong. |
| 86 | `1139:165` | ✓ | ✓ | **Correct** | The `Breakfast` surface token falls, heading/transition candidates rise, and breakfast options follow. |
| 87 | `1139:248` | ✓ | ✗ | **Incorrect** | A Lunch section is plausible, but `x`, hyphen, comma, and period do not increase. |
| 88 | `1193:99` | ✗ | ✗ | **Incorrect** | Amazon/Google are unsupported, and `Google` decreases despite being called encouraged. |
| 89 | `1193:173` | ✗ | ✗ | **Incorrect** | The suppression target and guessed countries/facilities are both unsupported. |
| 90 | `1193:414` | ✓ | ✗ | **Incorrect** | Innovation/product benefits fit, but `provide` is not the suppressed token. |
| 91 | `1234:64` | ✓ | ✓ | **Correct** | `More` falls, all named information/medical terms rise, and further case information fits. |
| 92 | `1234:65` | ✓ | ✗ | **Incorrect** | Image/case-link content fits, but `pictures` decreases rather than rises. |
| 93 | `1234:68` | ✓ | ✓ | **Correct** | `Details` falls, the named transition/link tokens rise, and more resources are plausible. |
| 94 | `1240:234` | ✗ | ✗ | **Incorrect** | The comic-page interpretation is unsupported and `I` decreases despite being called encouraged. |
| 95 | `1240:357` | ✗ | ✗ | **Incorrect** | The claimed suppression and the “late at home” narrative both miss the political comic. |
| 96 | `1240:478` | ✗ | ✗ | **Incorrect** | Digits remain top predictions but their logits decrease; the quiz-score interpretation is also false. |
| 97 | `1266:127` | ✓ | ✗ | **Incorrect** | Payment methods fit, but `a` and `the` decrease even though bank/credit terms rise. |
| 98 | `1266:336` | ✓ | ✗ | **Incorrect** | A monetary amount is right, but `0` is not the suppressed token; the dollar-sign family is. |
| 99 | `1266:377` | ✓ | ✓ | **Correct** | `additional` falls, fee/student/program terms rise, and the school-fee interpretation fits. |
| 100 | `1326:78` | ✓ | ✗ | **Incorrect** | The theological interpretation is strong, but `that` is not the salient suppressed token. |

## When the delta NLA is jointly correct

The 20 joint successes are:

`4, 8, 10, 13, 24, 26, 30, 31, 44, 49, 59, 69, 77, 78, 81, 83, 86, 91, 93, 99`.

They tend to combine three properties:

1. **A locally constrained boundary.** Fixed phrases, headings, lists, and narrow grammatical slots limit both the token movement and its interpretation—for example “ranks → of” (#49), `Breakfast` as a heading (#86), or an additional fee (#99).
2. **A coherent, narrow domain.** Router products (#24), research assurances (#26), cooperative members (#77), Brexit (#81), and medical case information (#91/#93) provide enough repeated signal to attach token movement to the right subject.
3. **Little semantic overreach.** The explanation stays near the observable transition rather than adding an unseen company, country, sport, or event.

## Three ways it is incorrect

### 1. Token-correct but semantically incorrect: 28

These examples show that the AV can decode real logit movement but attach the
wrong meaning. Examples include basketball becoming football (#51), an
international congress acquiring UNESCO/UN (#58), North America becoming
Europe/UK (#60), and a Street View scene becoming New York traffic (#73).

This is the clearest evidence that **directionally correct token information is
not sufficient for a faithful explanation**. Many unrelated tokens can receive
positive logit changes, and the AV can build a fluent story around them.

### 2. Semantically plausible but token-incorrect: 28

These examples infer the right topic or continuation type but state at least
one wrong literal change. Examples include a correct cycling transition with
punctuation that actually decreases (#45), a plausible market-number
continuation while `$` decreases (#52), correct musical-constraint semantics
while `rules` slightly decreases (#64), and correct payment semantics while
`a`/`the` decrease (#97).

This suggests that some explanations are **contextual paraphrases**, not
precise reports of the delta.

### 3. Incorrect on both: 24

These concentrate around partial words, timestamps, dates, noisy web text, and
the one generation collapse: `03:` (#11), `cr…` in “crabcake” (#71), the
`2002-2…` date (#80), the invalid multilingual/code output (#75), and digits
after “Share June” (#96).

These boundaries require composing subword or formatting information into a
larger lexical object. The checkpoint often treats the raw token movement as a
semantic explanation instead.

## Bottom line

Under the joint definition, this sample supports a **20% explanation-correctness
rate**, not 48%. The earlier semantic-only and token-only counts were each 48%
by coincidence: only 20 examples belong to both sets. The model is most
reliable when syntax, token movement, and document topic all point to the same
narrow continuation. It is least reliable when any one of those signals is
ambiguous—especially referential details, proper nouns/geography, or fragmented
numeric and subword boundaries.
