```
Elias Stengel-Eskin* 1 Archiki Prasad* 1 Mohit Bansal^1
```
## Abstract

```
While large language models (LLMs) are increas-
ingly being used for program synthesis, they lack
the global view needed to develop useful ab-
stractions; they generally predict programs one
at a time, often repeating the same functional-
ity. Generating redundant code from scratch
is both inefficient and error-prone. To address
this, we proposeRefactoring forGeneralizable
AbstractionLearning (REGAL), a gradient-free
method for learning a library ofreusablefunc-
tions via coderefactorization, i.e., restructuring
code without changing its execution output.RE-
GALlearns from a small set of existing programs,
iteratively verifying and refining its abstractions
via execution. We find that the shared function
libraries discovered byREGALmake programs
easier to predictacross diverse domains. On five
datasets – LOGO graphics generation, Date rea-
soning, TextCraft (a Minecraft-based text-game)
MATH, and TabMWP – both open-source and
proprietary LLMs improve in accuracy when pre-
dicting programs withREGALfunctions. For
CodeLlama-13B,REGALresults in absolute ac-
curacy increases of 11 .5%on LOGO, 26 .1%on
date understanding, and 8 .1%on TextCraft, out-
performing GPT-3.5 in two of three domains. Our
analysis revealsREGAL’s abstractions encapsu-
late frequently-used subroutines as well as envi-
ronment dynamics.^1
```
## 1. Introduction

```
An increasing range of tasks can be tackled by using a large
language model (LLM) to generate an executable program
for a given query; this paradigm has been applied in com-
puter vision (Sur ́ıs et al., 2023; Gupta et al., 2018; Cho et al.,
```
*Equal contribution (^1) UNC Chapel Hill. Correspondence to:
Elias Stengel-Eskin<esteng@cs.unc.edu>.
Proceedings of the 41 stInternational Conference on Machine
Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by
the author(s).
(^1) Code: https://github.com/esteng/regal_
program_learning.
**Q:** A **small 9 gon** and a
**Q:** smallA **small 9 gon** to the left
of a small 5 gon for i in range( 9 ):
forward(left(40.0 2 ))
penup();forward( pendown() 8 )
for i forward(in range 2 )( 5 ):
left(72.0)
for forward( j in range 4 )( 6 ):
#Incorrect reasoning for i in range( 9 ):
forward( **left(40.5** 2 ) **)** #Math error
left(60.0)
**Q:** 6-sided snowflake with a
line and **small 9 gon** as arms
**Q:** 6 sided snowflake with a
line and **small 9 gon** as arms
for j # Correct reasoningin range( 6 ):
(^) **draw_small_9gon** embed('forward(4)()',#Reuse
(^) left(locals60.0()))
**ReGAL:** Discovers abstractions that
can be **reused** as helper functions
**draw_small_5_gon** ()
**draw_small_9_gon** () **Code
Bank**
generating programs from scratch
**draw...
draw...**
fetch
helpers
Figure 1.REGALtrains by refactoring primitive-only programs
into abstractions that are verified and stored. These abstractions
have two benefits:Reusability: Rewriting the same code multiple
times leads to errors;Abstraction: REGALmakes prediction eas-
ier by allowing matching between the query and the abstractions.
2023), robotics (Ahn et al., 2022; Singh et al., 2023), tool
use (Schick et al., 2023; Lu et al., 2023; Qin et al., 2023),
and complex reasoning (Lyu et al., 2023). In all these cases,
the overall program generation framework is the same: an
individual query is given (along with an instructive prompt)
to an LLM, which produces a program that, when executed,
yields the desired result. Crucially, each program is gener-
atedindependently(as shown in Fig. 1), with no reference
to other queries or programs, and is composed ofprimitive
operations, i.e., the domain language’s built-in operations.
This approach has two major and related limitations:
1) Lack of Reusability: Each program is designed as a
one-off script to solve a given example but is not reused
by other examples. This increases redundancy and can re-
sult in unnecessary errors: for two examples requiring a
shared subroutine, the model might correctly generate the
subroutine in one example and make a mistake in the other.
For instance, in Fig. 1 (top) although the “primitive-only”
model had previously generated nonagons, it draws a poly-
gon with an incorrect angle.REGAL’sdrawsmall9gon()
function, on the other hand, executes correctly.
2) Lack of Abstraction: Shared abstractions can improve
accuracy by making skills more accessible to the model.
When generating from primitives alone, the model must
interpret the query and generate the correct mapping from
the query tomultiple primitives, requiring more reasoning.
The model’s overall task becomeseasierwhen it uses in-
terpretable abstractions, as it is choosing a function name

# arXiv:2401.16467v2 [cs.SE] 6 Jun 2024


from a library instead of reasoning from scratch. In Fig. 1
(bottom) a model augmented with abstractions can match
the sub-query“a small 9 gon”todrawsmall9gon(); with
this part of the task simplified, the model reasons correctly
about the remaining code, while the primitive-only model
fails to correctly embed the shape in a loop.

Both limitations can be traced to alack of global contextas
the model sees each example separately, so it lacks a mech-
anism for developing reusable abstractions. This differs
greatly from how humans write code: generally, developers
might start solving individual tasks with one-off solutions,
but quickly begin to develop a library of shared abstractions
and code snippets for related problems, thereby reducing re-
dundancy in their code, promoting efficiency and readability
(McConnell, 2004; Downey, 2012). Furthermore, functions
can beverified: once we have tested a function, we can
rely on it in the future – something that is harder to do for
ever-changing one-off code snippets. Such abstraction and
verification is only sensible if the code synthesis process
takes place over the course of multiple examples. In other
words, if presented with a single, one-off task, there is no
reason not to write a one-off script.

While abstraction offers numerous benefits, it comes with
the risk of over-fitting, where a function tailored to a spe-
cific example loses its generalizability. For instance, in
Fig. 1, a function likedraw9gonsnowflake()may per-
fectly match one example but fails to generalize. Conversely,
drawsmall9gon()is a more versatile function applicable
in various contexts. The ability to produce novel programs
using primitive operations needs to be balanced with the
benefits of encoding subroutines into reusable abstractions
(O’Donnell, 2015). A similar balance between flexibility
and efficiency appears in a variety of domains, including
language (O’Donnell, 2015; Yang, 2016), biology (Futuyma
& Moreno, 1988), manufacturing (Flynn & Jacobs, 1987),
and programming (Ellis et al., 2021). To strike this balance
in LLM-based program synthesis, we proposeRefactoring
forGeneralizableAbstractionLearning (REGAL). REGAL
refines abstractions iteratively by refactoring programs as
well as verifying, correcting, and pruning abstractions such
that overly specific or incorrect programs are improved upon
or removed.REGALrelies on two key elements: a small
set of programs using primitive operations (i.e.,primitive
programs) and an execution environment (e.g., Python). Im-
portantly, we showREGALcan learn from LLM-generated
programs without requiring any human annotations.

REGALfollows a familiar train-test paradigm: duringRE-
GAL’s modular training phase (see Fig. 2), it iteratively
refactors a small set of(query, program)examples to pro-
duce a library of useful abstractions.REGALuses an LLM
to write helper functions for a batch of examples, which
are verified against the expected result; successful helper

```
functions are added to the library and the refactored pro-
gram serves as an example of the function’s usage.REGAL
can take success feedback into account to correct and debug
errors, and it periodically edits the helper functions to make
them more generalizable or – if they cannot be made more
generic – prunes functions that are overly specific. Note
that the training is gradient-free, relying on a frozen LLM
to refactor programs. In the testing phase, an LLM agent
is tasked with predicting programs for test queries. The
agent has access toREGAL’s library of helper functions
and demonstrations of how to use them.
We demonstrate the broad applicability of REGAL by test-
ing it on five diverse datasets: LOGO (Ellis et al., 2021;
Wong et al., 2021), a program induction task; a date rea-
soning task (Srivastava et al., 2022) known to challenge
LLMs (Suzgun et al., 2022); TextCraft (Prasad et al., 2023),
a text-based game for crafting Minecraft objects; a sub-
set of MATH (Hendrycks et al., 2021) which contains al-
gebra word problems, and TabMWP (Lu et al., 2022), a
collection of math quesitons about tables. Across these
tasks,REGALsignificantly improves the accuracy of the
predicted programs from various LLMs – especially open-
source LLMs – over a baseline that predicts primitive pro-
grams (i.e., programs without REGAL’s abstractions). For
instance, CodeLlama-13B’s (Roziere et al., 2023) accuracy
increases by 11 .5%, 26 .1%, and 8 .1%on LOGO, Date,
and TextCraft respectively, surpassing larger models like
GPT-3.5 (cf. Sec. 5). In Sec. 6, we show thatREGAL’s
abstractions are reusable across examples, encapsulate key
domain functionalities, and we include an error analysis fur-
ther highlighting the features that makeREGALeffective.
Sec. 6.3 reveals thatREGALcan improve over baseline
primitive programs with minimal examples, yielding major
improvements even with a50%reduced training set.
```
## 2. Related Work

```
Program Induction. Program induction involves learning
a symbolic and programmatic mapping of inputs to outputs.
Humans are adept at this kind of “rule-learning” (Marcus
et al., 1999; Furnkranz et al., 2012). ̈ REGALalso aims
to learn a set of general functions that can be used to map
inputs to outputs, i.e., a form of program induction. Ellis
et al. (2021) present DreamCoder, a method for combining
program induction and synthesis that uses a wake-sleep
Bayesian learning method to learn programs. Wong et al.
(2021) extend this work to incorporate language, using an
alignment model as part of the joint model. Like Ellis et al.
(2021), Grand et al. (2023) adopt a similar symbolic search
procedure, but use an LLM to document abstractions. The
symbolic search procedure used by this line of past work has
relied on data structures that assume the domain language
isλ-calculus (Lake et al., 2015; Ellis et al., 2021; Wong
et al., 2021; Grand et al., 2023), which is not typically used
```

for software development. In contrast,REGALhas an
LLM-based search procedure, allowing us to use flexible
languages like Python, which are more commonly used by
developers and better represented in pre-training data.

Program Synthesis and Tool Use. Tool use by LLMs
(Schick et al., 2023; Mialon et al., 2023) refers to a form of
program synthesis or semantic parsing where an LLM gen-
erates API calls to external tools (e.g., calculators, search
functions, etc.). This formulation has also been applied
to reasoning tasks (Lyu et al., 2023; Chen et al., 2022) as
well as other domains such as computer vision (Sur ́ıs et al.,
2023; Gupta & Kembhavi, 2023; Cho et al., 2023), summa-
rization (Saha et al., 2022), and robotics (Ahn et al., 2022;
Singh et al., 2023; Huang et al., 2022; 2023). Past works
have attempted to induce tools from examples. Cai et al.
(2023) induce tools using an LLM for reasoning tasks from
BigBench (Srivastava et al., 2022); unlike our work, their
system generates one tool per task. While this can offer ben-
efits for homogenous reasoning tasks (e.g., sorting words
alphabetically), heterogenous tasks like the ones we explore
require multiple functions. More akin to our work, Yuan
et al. (2023), Qian et al. (2023), and Wang et al. (2024)
induce multiple tools for vision and math tasks using an
LLM-based framework which also includes retrieval-based
parsing. In addition to focusing on different domains, we
place an emphasis on learning a shared code bank that can
be used by multiple LLMs to generate programs including
several open-source LLMs. We also differ in our focus on
refactoring, and in the amount of information we provide to
the refactoring model: unlike Yuan et al. (2023) and Qian
et al. (2023), we do not provide in-context examples of the
kinds of tools we want the model to create, investigating
instead what abstractions the model builds without domain-
specific guidance. While Wang et al. (2024) also prunes
their learned functions, their method does not involve refac-
toring; in contrast, we learn functions by refactoringand
periodically edit the codebank (in addition to pruning it).

Induction in Interactive Domains. Wang et al. (2023)
also induce functions in a Minecraft domain; however, theirs
are written and stored based on one iteration. In contrast,
our work refactors programs in a group and tests and refines
them across the training process, showing generalization
in multiple domains. Other prior work learns a library of
abstractions for planning in embodied domains (Wong et al.,
2023; Majumder et al., 2023). While we share a similar
motivation,REGALoperates in the space of generating ex-
ecutable programs instead of PDDL operators (Wong et al.,
2023) or causal textual feedback (Majumder et al., 2023).
Similarly, our work aligns with prior efforts in LLM-based
task decomposition (Khot et al., 2023; Prasad et al., 2023),
where skills are reused across multiple task instances. How-
ever, these approaches manually identify atomic skills and
require the LLM to repeatedly execute skills from scratch. In

```
contrast,REGALprovides a way of automatically discover-
ing such abstractions and reusing them via helper functions.
```
## 3. Methodology

```
In this section, we describe the overall pipeline of
our method: Refactoring forGeneralizableAbstraction
Learning (REGAL). REGALconsists of two phases: the
trainingor induction stage where abstractions (i.e., helper
functions) are learned, and thetestingor synthesis stage,
where abstractions are used to generate programs for test
queries. During training, as illustrated in Fig. 2,REGALdis-
covers reusable abstractions by generating candidate helper
functions, validating their correctness, and debugging via
editing and pruning of ineffective helper functions. Given a
set of demonstrations(q,p)of queriesqand gold primitive
programsp, we first preprocess the data to cluster examples
based on query similarity, described in Sec. 3.1. The train-
ing stage then builds abstractions by refactoring primitive
programs in batches (Sec. 3.2), while the testing stage solves
new queries by generating programs that glue together the
learned abstractions with primitive operations (Sec. 3.3).
We use GPT-3.5 for training; at test time we use a range of
LLMs, focusing on freely available open-source LLMs.
```
```
3.1. Preprocessing
Before training, we preprocess queries and programs(q,p)
by (optionally) adding comments, clustering them into re-
lated batches, and sorting them by approximate difficulty.
Adding Comments. We optionally add comments to align
the query with the primitive program, enabling the model
to generate the correct abstractions. We present each(q,p)
pair to GPT-3.5 independently, with a prompt asking the
model to commentpbased onq; we then verify that the
commented code executes to the same result.
Clustering Data. In order to form abstractions that are
sharedbetween examples, the refactoring LLM requires
a multi-instance scope, i.e., it must receive a batch of re-
lated(q,p)tuples at a time. We implement this by cluster-
ing examples using an embedding of the queryq. Specifi-
cally, we embed each query using OpenAI’s Ada embedding
model (OpenAI, 2022) and hierarchically cluster the embed-
dings using Ward’s clustering algorithm (Ward Jr, 1963),
implemented via Scikit-Learn (Pedregosa et al., 2011). This
gives us a tree of related examples, which we topologically
sort and group into batches ofk, wherekis a hyperparame-
ter (see Appendix C for all hyperparameter values).
Curriculum. Intuitively, shorter and easier programs
should contain abstractions that can be reused in harder,
more compositional programs, so we sort examples into a
curriculum (Bengio et al., 2009). To approximate difficulty,
we sort the batches based on the average length (in tokens)
of their queries. See Appendix A.1 for preprocessing details.
```

```
Stage 3(b): pruneCodeBank()
```
```
Q: craft yellow wool
```
```
inventory = check_inventory()
get_object('8 terracotta')
inventory = check_inventory()
get_object('1 dandelion')
craft_object('1 yellow dye',
['1 dandelion'])
craft_object('8 yellow
terracotta',['8 terracotta',
'1 yellow dye'])
```
```
Q: craft yellow terracotta
```
```
Q: craft light blue terracotta
```
```
Stage 1: refactorBatch()
```
```
Primitive Programs
```
```
Prompt: Refactor this program
with reusable helper functions
```
```
def get_ingredient(ingredient):
def craft_and_get_ingredient(target,
ingredients):
for ingredient in ingredients:
...
return inventory
helper
functions Q: craft yellow wool
```
```
craft_and_get_ingredient('
yellow dye',
['1 dandelion'])
craft_and_get_ingredient('
yellow terracotta',
['8 terracotta', '1 yellow
dye'])
```
```
Q: craft yellow terracotta
```
```
Q: craft light blue terracotta
```
```
Refactored
Programs
```
```
Prompt: The following
programs failed with
error message...
```
```
def ...
```
```
Code Bank
```
```
Demo Bank
```
```
add verified
helpers
```
```
Stage 2a: verify()
```
```
Stage 2b: retry()
```
```
Prompt: Helper function
craft_and_get_ingredient ()
is passing 2/3 times, please
improve it.
```
```
Stage 3(a): editCodeBank()
```
```
def ...
Prune functions with many failures
```
```
Edit existing functions
to improve passing rate
```
```
def ...
```
```
Code Bank Demo Bank
```
```
ReGAL:
```
```
add examples of
helper usage
```
Figure 2.REGALstarts by refactoring a batch of primitive programs to develop a set of modified programs and helper functions (Stage
1 ). It then verifies the results of refactored programs, optionally retrying failed programs according to environment feedback. Useful
helper functions are added to the Code Bank along with example usage added to the Demo Bank (Stage 2). Periodically, we edit and
prune the Code Bank to improve its functions (Stage 3). At test time, theREGALagent has access to the Code Bank, the Demo Bank,
and the remaining original programs. It is compared against a baseline agent which has access to a larger number of original programs.

3.2. Training

REGAL’s training data consists pairs of queriesqand prim-
itive programsp. The training phase outputs: 1) theCode
Bank(C): the library of helper functions abstracted out
during training and 2) theDemo Bank(D): examples of the
functions being used. As shown in Fig. 2, the training phase
is an iterative process where the LLM receives as input a
batch of queries and primitive programs and then proposes
helper functions that can be abstracted (Stage 1). For each
batch, candidate helper functions are evaluated based on the
correctness of the refactored programs that occur in (Stage
2a). After verification, failed examples are isolated into a
second batch and re-attempted (Stage 2b). To improve the
quality of the Code Bank, we periodically edit helper func-
tions after a fixed number of batches to improve their pass
rate (over unit tests) and prune ineffective helpers (Stage 3).
After training, the library of helper functions (Code Bank
C) is stored for use during testing along with successful
demonstrations of programs using helper functions (Demo
BankD). Note that the training process can be repeated over
multiple epochs. Below we describe each stage in detail,
with the overall algorithm detailed in Algorithm 1.

Stage (1): Refactoring Examples ( **refactorBatch** ).
The main module of the refactoring process takes as input
a batch of examples, a set of instructions, and the current
set of helper functions in the code bank (if any). It prompts
the refactoring LLM for a set of new helper functionsHnew
along with refactored versions of each program that uses
helpers fromHnewwhen appropriate.

Stage (2a): Verification ( **verify** ). To avoid introducing
errors, we need to verify the helper functions and refac-
tored programs generated by the LLM by executing them
and comparing the results to the original, i.e., determining

```
ifpˆ()=p(). The refactored program(q,pˆ)is stored as
a demonstration for future use by the agent (cf. Sec. 3.3)
if it passes verification. Only helper functions that pass
verification are added toC. We also store a record of pro-
grams thatfailedverification, as these will be crucial in
editCodeBank()andpruneCodeBank(), which improve
existing functions and prune functions leading to failures.
Stage (2b): Feedback-based Retrial ( retry ). If a pro-
gram fails to pass verification, we optionally retry the refac-
toring process. In a follow-up prompt, we present failed
programs and their helper functions. We also include envi-
ronment feedback from execution (i.e., the output or error
message produced).^2 The refactoring LLM then produces a
new version of each failed program; these are verified and
their helpers are added toCif correct.
Stage (3a): Editing Code Bank ( editCodeBank ).
From theverify()module, some helper functions fail to
pass all unit tests because they contain incorrect abstractions.
For example, a function likedrawtriangle()might start
with a hardcoded value for a small size, leading it to fail on
medium triangles. To update such functions, we construct a
prompt for each function inDthat shows the LLM passing
and failing unit tests and asks it to propose edits to the func-
tion; this occurs once everyeditEveryiterations, where
editEveryis a hyperparameter. We replace a function if it
passes more unit tests after editing.
Stage (3b): Pruning Code Bank ( pruneCodeBank ).
In this module, we prune helper functions added toCthat
fail a majority of unit tests and cannot be improved further
via editing. For each function, we derive a score based on
the success rate of programs using the function; we set a
```
(^2) We do not include the output for LOGO as it is an image.


threshold below which functions are pruned (shared by all
domains). See Appendix A.2 for further details.

We use the dev set to select hyperparameter values, reported
in Appendix C. All prompts can be found in Appendix D.

3.3. Testing

At test time, we deploy a program synthesis – or semantic
parsing –agentthat makes predictions for test examples,
one at a time. Unlike related work on using learned tools
(Yuan et al., 2023; Qian et al., 2023; Wong et al., 2023),
we explore a variety of open-source LLMs, in addition to
a black-box LLM (GPT-3.5). Following effective strate-
gies in semantic parsing and in-context learning (Shin &
Van Durme, 2022; Roy et al., 2022; Bogin et al., 2023; Liu
et al., 2022; Yasunaga et al., 2023), for each test example,
the agent constructs a prompt with in-context learning (ICL)
examples retrieved from a training corpus, followed by a
test query. The examples are retrieved from the training data
using vector similarity between the training queries and the
test query. Further details in Appendix A.3.

REGAL-augmented Agent.Our agent has access to the
training dataandcode bank, as well as examples of refac-
tored programs in thedemo bank. The ICL budget (
examples for all experiments) is split between primitive
training examples and refactored ones.^3 In addition to these
demonstrations, the augmented agent retrieves up to 20 rele-
vant helper functions from the code bank, where relevance
is measured by the similarity between the query and the
function name and description. These helper functions are
concatenated into the prompt. The final input is a prompt
containing the instructions, the retrieved helper functions,
the mixed ICL examples, and the test query. To encourage
the model to use helper functions, we include a ReAct-style
prompt (Yao et al., 2023) that first asks the model tothink
about which functions might be relevant based on the query
and then generate the code.^4

## 4. Experimental Setup

4.1. Datasets

We explore five datasets: LOGO, Date understanding,
TextCraft, MATH, and TabMWP. A common thread through
these datasets is that they contain heterogenous problems
requiring multiple helper functions as opposed to problems
like sorting, which are challenging for LLMs but can be
solved with a single function (Dziri et al., 2023). Statistics
for the datasets are given in Table 5. See Appendix A.5 for

(^3) We found it necessary to keep some primitive programs as ICL
examples, as not all test queries can be handled by helper functions
alone. We treat the ratio of primitive to refactored programs in the
ICL example as a hyperparameter (all values listed in Table 9).
(^4) Without these additional “thought” statements, we found the
augmented agent rarely uses any helper functions.
details about each dataset and its primitive operations.
LOGO. LOGO is based on the Logo Turtle graphics
domain-specific language (Abelson & DiSessa, 1986), with
which basic graphics can be drawn by controlling a pen
(the “turtle”) that draws as it moves through space, using
commands likeforward(dist)andleft(theta). The
data we use is based on Ellis et al. (2021)’s LOGO dataset,
re-annotated by Wong et al. (2021). For easier prediction
by LLMs, we parse the data into abstract syntax trees and
write a set of rules for translating these into Python; we
release this rewritten data. We use the “small” train/test
splits (200/111) from Wong et al. (2021) and take 100 dev
examples from the “large” train set.
Date. We use the date understanding task from BigBench-
Hard (Srivastava et al., 2022; Suzgun et al., 2022), which
consists of short word problems requiring date understand-
ing. We obtain silver programs from Lyu et al. (2023)’s
predictions. Specifically, we split their predicted programs
from GPT-3.5 into train, dev, and test splits (66/113/180)
and filter the train split by correctness.
TextCraft. To explore the utility ofREGALin LLMs-
as-agent settings (Liu et al., 2023), we use the TextCraft
dataset (Prasad et al., 2023) that requires an agent to craft
Minecraft items within a text-only environment (Cˆote et al., ́
2019). Each task instance in TextCraft consists of a goal
(query) and a series of 10 crafting commands that contain
recipes for related items including distractors. Unlike Prasad
et al. (2023), who use TextCraft in an interactive setting
where the LLM agent receives textual feedback from the
environment at each step, we ask the agent to generate a
single Python program for executing the entire task at once,
making the taskmore challenging. We evaluate on the depth
2 split of the test set used in Prasad et al. (2023) while using
a subset of depth 1 recipe examples for our dev set, giving
us a train/dev/test split of 190/50/77.
MATH. To testREGAL’s ability to discover functions in
domains that donothave a pre-defined domain language, we
additionally examine the MATH dataset (Hendrycks et al.,
2021) consisting of challenging math word problems. As
with Date, we generate silver training programs; in this
case, we use GPT-4 with a Program-of-Thoughts prompt
(Chen et al., 2022). To have a good chance of generating
sufficient numbers of training programs, we use examples
from hardness levels 1 and 2 (MATH contains 5 levels of
difficulty). We also focus on the Algebra subset of MATH,
which is the largest. This gives us a train/dev/test split
of 194/61/74. While only correct programs are used for
training, correct and incorrect examples are used in dev
and test. Note that unlike the other datasets, MATH does
not have an inherent domain language associated with it
and does not have any primitives beyond the ones already
contained in Python.


Table 1.Accuracy of baseline agents predicting primitive programs (Prim.) and those augmented withREGALhelper functions (3 random
seeds). Across domains and models,REGALimproves over a strong baseline agent with access to the same number of ICL examples.
Math domains with no clear domain language marked with∗.

```
LOGO Date TextCraft MATH (Alg.)∗ TabMWP∗
Agent Prim. REGAL Prim. REGAL Prim. REGAL Prim. REGAL Prim. REGAL
```
CodeLlama-7B 34. (^5) ±1.3 34. (^5) ±1.6 52. (^4) ±0.7 55. (^2) ±1.4 12. (^8) ±1.3 16. (^7) ±1.3 6. (^8) ±0.8 17. (^6) ±0.8 16. (^7) ±1.6 27. (^7) ±0.
CodeLlama-13B 45. (^6) ±0.3 57. (^1) ±0.6 42. (^8) ±2.0 68. (^9) ±1.6 18. (^8) ±0.7 26. (^9) ±2.2 14. (^0) ±0.9 17. (^6) ±0.8 29. (^1) ±0.6 27. (^9) ±0.
CodeLlama-34B 50. (^2) ±0.8 50. (^8) ±0.6 47. (^2) ±1.5 68. (^5) ±2.1 22. (^2) ±0.7 30. (^8) ±1.3 14. (^9) ±1.1 22. (^5) ±1.6 19. (^4) ±1.2 27. (^0) ±0.
Lemur-70B 44. (^1) ±1.4 56. (^8) ±0.9 68. (^2) ±0.4 70. (^5) ±0.6 15. (^7) ±1.7 23. (^5) ±2.1 14. (^0) ±1.2 13. (^1) ±1.2 27. (^5) ±0.9 25. (^7) ±0.
GPT-3.5-turbo 36. (^9) ±1.6 49. (^3) ±1.1 88. (^9) ±0.3 90. (^2) ±0.5 15. (^4) ±1.3 18. (^4) ±2.0 43. (^2) ±1.6 55. (^4) ±2.8 87. (^4) ±0.9 89. (^2) ±0.
TabMWP. We further extend our general experiments on
MATH by testing on TabMWP (Lu et al., 2022), a tabular
resoning dataset consisting of math word problems about
tabular data. Here, we also use silver programs for training
and focus on levels 1-4 (TabMWP has 8 difficulty levels),
following a similar procedure as for MATH. This gives us a
train/dev/test split of 194/60/74.
4.2. Baselines
Baselines from Prior Work. We compareREGAL
against relevant external baselines from past work. However,
note that multiple methodological differences in our work,
like the use of ICL examples and the format of the output
programming language, give our agent an inherent advan-
tage over these baselines. Thus, we refer to these numbers
primarily to highlight the strength of our baseline agent. For
LOGO, we use the “offline synthesis” numbers reported by
Grand et al. (2023), which resemble our train/test setting;
however, we note that Grand et al. (2023) predict programs
in their original Lisp format and use a different agent model.
For the Date dataset, we run Lyu et al. (2023)’s Faithful-
CoT method on our custom test split usinggpt- 3. 5 - turbo.
While the output format and models used are the same,
both our baseline andREGALuse retrieved examples for
in-context learning, while Lyu et al. (2023) do not. Further-
more, our ICL examples are based on programs generated
by Lyu et al. (2023) after filtering for correctness, leading
to better performance even from our baseline agent. Finally,
for TextCraft we re-run Prasad et al. (2023)’s baseline –
based on ReAct (Yao et al., 2023) – on the depth-2 test set
of TextCraft. Here, we usegpt- 3. 5 - turbo-instruct, as
Prasad et al. (2023) found it to outperformgpt- 3. 5 - turbo.
Baseline Programming Agent. For a more direct com-
parison, we implement a baseline agent that has access to
all the same data asREGALbut does not use abstractions,
thus directly testing the role ofREGALabstractions in per-
formance. Our baseline agent retrieves primitive programs
from the training data; note that this is exactlythe same
datasetused for refactoring, i.e., the baseline LOGO agent
retrieves demonstrations from the LOGO training examples.
The input to the baseline agent is a prompt with the same
overall instructions as theREGALagent (including a de-
scription of the primitives), the ICL examples, and the test
query; the output is a program for the test query. We use a
fixed budget of 10 ICL examples so that the baseline agent
sees exactly as many demonstrations as the REGAL agent.

## 5. Results

```
Comparison to External Baselines.Table 2 shows a com-
parison of the baseline andREGALagents the external
baselines from prior work. Note that we do not include
MATH and TabMWP here as we use a subset of levels
for these datasets. We first note thatREGALoutperforms
the baselines in all cases. Furthermore, because of the
methodological differences detailed in Sec. 4, our baseline
“primitive-only” agent – equipped with ICL examples and
using a code-specific LLM – also outperforms past base-
lines on LOGO and Date. On TextCraft, the ReAct baseline
from Prasad et al. (2023) has an advantage in that it receives
environmental feedback, while our baseline does not. Never-
theless, evenwithout feedbackREGALoutperforms ReAct.
Thus, we compare primarily against our baseline agent, as
this provides a direct measure of the impact of abstractions
(rather than the other changes made).
REGALoutperforms the baseline agent in non-math
domains.Table 1 showsREGAL’s performance compared
to the baseline agent using primitive programs (described
in Sec. 4.2). Overall, for each model type,REGALgen-
erally outperforms the baseline by a wide margin; for ex-
ample,REGALprovides CodeLlama-13B a 11 .5%boost
on LOGO, allowing it to outperform much larger models.
Across datasets, CodeLlama-13B generally benefits most
fromREGALabstractions. Table 1 also shows that large
models also benefit fromREGAL, with large gains for
Lemur-70B and GPT-3.5 on LOGO and TextCraft. Finally,
```

Table 2.Comparison to relevant past work in each non-math do-
main. TC†denotes the TextCraft dataset.

```
Method Agent Acc.
```
```
LOGO
```
```
LILO Code-davinci 41. 1
Primitive Programs CodeLlama-13B 45. 6
REGAL Programs CodeLlama-13B 57. 1
```
```
Date
```
```
Faithful-CoT GPT-3.5-turbo 83. 3
Primitive Programs GPT-3.5-turbo 88. 9
REGAL Programs GPT-3.5-turbo 90. 2
```
```
TC
```
```
†ReAct GPT-3.5-turbo^25.^6
Primitive Programs CodeLlama-34B 22. 2
REGAL Programs CodeLlama-34B 30. 8
```
the largest models are not necessarily the best: on LOGO
and TextCraft, GPT-3.5 is outperformed by open-source
models, especially after augmentation, e.g., CodeLlama-
13B withREGALabstractions is able to outperform GPT-
3.5withoutabstractions by 19 .2%(it also outperforms GPT-
3.5 with abstractions). Thus, by runningREGAL’s training
process on only∼ 200 examples, we can bring a much
smaller open-source model’s accuracy far beyond that of a
(likely) much larger system.

REGALimproves over the baseline on mathematical
domains.Our math domains (MATH and TabMWP) differ
from the others in that there is no clear domain language
thatREGALshould discover. Unlike a domain like LOGO
(where it is clear which functions a human programmer
would write), there is less clarity on which helper functions
should be discovered. Nevertheless, in this challenging set-
tingREGALgenerally improves over the baseline agent
across different models, with major gains for GPT-3.5 on
MATH ( 11 .8%) and for CodeLlama-7B on TabMWP (11%),
where the 7BREGALperformance is comparable to the
13B performance. Overall, Lemur struggles with these prob-
lems, producing a large number of empty outputs both for
the baseline andREGALagents; when taking the standard
error into account, baseline andREGALperformance is
roughly comparable. These results highlightREGAL’s flex-
ibility and ability to generalize to domains without clear-cut
domain languages.

Ablations.In Sec. 3.2 we describeREGAL’s multiple com-
ponents; here we determine the utility of each by removing
each one in isolation for non-math datasets. Table 3 shows
the results of these ablations. We use the CodeLlama-13B
agent due to the size ofREGAL’s impact on it across tasks.
We average the performance across 3 seeds. Table 3 shows
that each component contributes to performance, with drops
when any is removed. Across datasets, the largest perfor-
mance decreases come with the removal of retrials and with

```
Table 3.Ablations of each optionalREGALcomponent tested on
dev splits with CodeLlama-13B. To remove the curriculum, we
randomly shuffle example clusters instead of presenting them in
order of shortest query to longest query.
```
```
Ablation LOGO Date TextCraft
REGAL 55.0 77. 0 34. 12
```
- retry 48 .3 51. 9 30. 43
- curriculum 36 .3 56. 6 28. 78
- pruneCodeBank 52 .0 65. 5 25. 26
- editCodeBank 53 .3 69. 6 27. 35

```
removal of the curriculum. Retrials can not only increase the
number of useful helper functions but can also help increase
the number of examples in the Demo Bank. Replacing
the curriculum with a random ordering also severely hurts
performance, e.g., leading to an 18 .7%drop on LOGO.
REGALlearns general, reusable functions.In Sec. 1, we
stressed the importance of reusability. Specifically, gener-
ating programs without shared abstractions means that the
model has to re-generate subprograms that could be reused
across multiple test instances. We argue thatREGALim-
proves over this paradigm by learningsharedabstractions.
The results in Table 1 indicate thatREGALoffers large
improvements over a baseline agent that lacks abstractions.
Here, we verify that the abstractions learned are reusable,
i.e., shared. Fig. 3 shows the number of times the top-5 most
commonREGALfunctions are called in test programs pro-
duced by the CodeLlama-13B agent. Across all datasets,
we see that the helper functions learned byREGALare
commonly reused, with the most relative reuse in TextCraft.
Appendix B.1 shows examples of these common functions.
```
## 6. Analysis

```
6.1. What kinds of programs are discovered?
```
```
To further examine what kinds of helper functions are dis-
covered, we examine the most frequent helper functions for
each domain from Fig. 3, summarizing the results below.
Refer to Appendix B.1 for the implementation of these func-
tions. We find that distinct trends emerge across domains.
For LOGO,REGALdiscovers functions that encapsulate
differenttypes of shapes. This is expected, as the LOGO
data was generated with these functionalities in mind, i.e.,
the larger shapes are composed of objects like semicircles,
pentagons, and circles. For Date,REGALtends to encapsu-
late single operations, prioritizinginterpretablity in function
nameslikegetdateoneyearago(). While seemingly
less complex than LOGO’s functions, this approach aligns
with the importance of function naming in synthesis proce-
dures, as highlighted by Grand et al. (2023). In TextCraft,
```

```
0 5
```
```
draw_med_semicircle
draw_sm_5gon
draw_sm_circle
draw_med_circle
draw_med_square
LOGO
```
```
0 5 10
```
```
get_date_tdy
get_date_1_week_ago
get_date_1_week_fr_tdy
get_date_1_year_ago
get_formatted_date
```
```
Date
```
```
0 20
```
```
craft_obj_with_ingr
check_and_get_obj
craft_with_ingr
craft_item
TextCraftget_obj_fr_env
```
Figure 3.Function usage by CodeLlama-13B for the top-5 most
common helpers illustrating reusability across examples. The x-
axis denotes the number of times a functions is used in the test set.

the functions uncovered byREGALare morecomplexand
reflect thedynamics of the game. Specifically, the functions
include conditional statements for checking ingredients, re-
flecting the fact that in TextCraft, having the correct craft-
ing ingredients is a prerequisite for crafting an object (see
craftandgetingredient()in Fig. 2 and Fig. 7, which
is taken from the learned code bankC).

6.2. What kinds of errors do agents make?

To better understand howREGALaids program generation
and also examine cases where it does not help, we perform
a two-part error analysis. First, we examine cases where the
REGAL-augmented program was correct and the baseline
agent’s primitive program was incorrect. We then examine
the opposite set of cases, where the baseline program was
correct but the REGAL program was incorrect.

Fig. 1 shows the first kind of comparison on LOGO using
the CodeLlama-13B model, where we qualitatively show
an actual example that highlights the benefits of reuse and
abstraction. The baseline program makes an error in cal-
culating the polygon’s interior angle when generating the
program from scratch. This is avoided by theREGALagent,
which simply uses a verified helper to generate the polygon
correctly. The example also illustrates the importance of
abstraction: as queries become more complex, generating
a solution from scratch becomes more challenging. The
baseline program reasons incorrectly about code outside
of the shape, failing to useembed()correctly. Meanwhile,
the augmented program offloads reasoning about the shape
to an easily-matched function, and is able to correctly use
embed(). To quantify these trends, we manually inspect the
output of the baseline CodeLlama-13B on LOGO on the 25
cases where theREGALagent was correct, categorizing

```
25 50 100 150
Size of Training Data
```
```
0
```
```
10
```
```
20
```
```
Success Rate
```
```
Primitive Programs ReGAL Programs
```
```
Figure 4.REGALprograms yield a higher success rate (accuracy)
compared to primitive programs on TextCraft for different sizes of
training setXusing CodeLlama-13B.
```
```
them into errors involving reasoning (first example in Fig. 1)
and shape-internal errors (second example); we find 16 rea-
soning and 9 shape errors. We also examineREGAL’s
failure modes by manually inspecting all cases where the
augmented agent failed and the baseline succeeded, again
using CodeLlama-13B on LOGO; there are 13 such cases.
We categorize them into three types:
```
- Incorrect connector code: (7 exs.) the program fails due
    to mistakes in the primitive operations or control flow.
- Incorrect/undefined function: (4 exs.) the code refers
    to non-existent functions, or incorrectly calls a function
    similar to the correct one.
- Verification failure: (2 exs.) the program was correct but
    the verification function gives a false negative.^5
Thus, the most common error is a failure to predict primi-
tives; here, theREGALagent is at a disadvantage w.r.t. the
baseline, as they share the same ICL budget. The baseline
agent sees 10 examples withonlyprimitive code, while the
    REGALagent sees 5 primitive and 5 Demo Bank examples.

```
6.3. How many training examples does REGAL need?
```
```
As mentioned in Sec. 4.2, both baseline andREGALagents
rely on demonstrations of queries and gold programs (X) to
retrieve most similar ICL examples. Additionally,REGAL
uses the same demonstrations to learn helper functions in
the code bankC. We now study how the performance of
both agents scales with the size of annotated gold programs
in train setXusing the CodeLlama-13B model on TextCraft.
From Fig. 4, we observe that theREGALagent consistently
outperforms the baseline agent as we vary the number of
training examples. Notably, helper functions learned by
REGALyield a 2 .56%improvement with as few as 25
demonstrations and an 8 .45%improvement with nearly half
the size of the train set used in Sec. 5. Additionally, the per-
```
(^5) For example, for“a small square next to a small 6 gon”the
agent generates the hexagon to the left of the square, where in the
reference it is to the right.


Table 4.Performance of baseline andREGALagent under distri-
bution shift on TextCraft with unseen objects in the test set.

```
Method Adaptation Setting Acc.
Prim. Programs − 7.
REGAL − 11.
REGAL pruneCodeBankon dev (unseen) 14.
REGAL All REGAL stages on dev (unseen) 17.
```
formance of both the baseline andREGALagents improves
as the number of demonstrations increases. This is expected
as both agents benefit from the retrieval of demonstrations
similar to the test query as the training set becomes larger
and consequently more diverse.

6.4. How can REGAL adapt to distribution shifts?

To assess the usefulness of abstractions learned byRE-
GALunder a distribution shift at inference-time, we use the
TextCraft dataset. We redistribute the instances into three
splits: train (seen), dev (unseen), test (unseen) consisting of
100/20/76 instances, such that the dev and test sets contain
objectsnever encounteredduring training. For example,
if the test set requires crafting “yellow jungle fence”, the
train set does not contain any instance with a “fence” as
a target object. Furthermore, we study how to adapt the
learned library with unseen examples from the dev set using
REGALin two ways: (i) only pruning the learned library
to retain generalizable functions (Stage 3(b) ofREGAL);
and (ii) an additional iteration of REGAL training.

The results are presented in Table 4. First, we observe
that the performance of both baseline andREGALagents
degrades under distribution shift because the in-context ex-
amples may not be adequately relevant to the test instance.
However, ReGAL outperforms the baseline agent with prim-
itive programs by 3 .6%. Furthermore, simply pruning an
existing code bank increases performance of ReGAL agent
by an additional 3 .5%. Note that this still uses a learned
codebank from prior training and without proposing any
new abstractions. Finally, learning new abstractions and
editing existing ones based on unseen dev data is the most
effective yielding an additional 2 .3%over only performing
prunceCodeBankon the unseen data for a total of 9 .4%
improvement over the baseline.

## 7. Discussion

Fixed vs. Variable Costs. In Sec. 5,REGALwas es-
pecially effective for open-source LLMs like CodeLlama.
This result is encouraging, as it indicates that we can bring
freely available and open-source models up to at least the
same performance as a proprietary, closed-source model (if
not more) usingREGALabstractions. Thus, we can convert
a variable cost – running an LLM on test data, which scales

```
linearly with the size of the test set – into the fixed cost of
running REGAL to learn a library of helper functions.
Connections to Semantic Parsing. Executable semantic
parsing (Winograd, 1972; Zelle & Mooney, 1996) typically
involves mapping queries to a domain-specific language
(DSL), a set of abstractions for a particular application, e.g.,
SQL operations for querying databases. These DSLs are
manually defined; one way to viewREGALis as a way
oflearninga DSL on top of an extremely general set of
primitives. A key benefit ofREGALis its generality: on
five different domains, it learns useful abstractions without
human intervention, while, in a standard semantic parsing
setting, these abstractions would designed by hand.
Connections to Hierarchical Reinforcement Learning.
Another way to view the functions discovered byREGAL
is as low-level policies composed of primitive actions. In
this view,REGALresembles hierarchical reinforcement
learning (HRL; Barto & Mahadevan, 2003), where tasks are
split into skill selection by a controller and the skill policies
themselves. Our agent LLM acts as a controller whileRE-
GAL’s training stage is responsible for discovering a useful
set of skills; this is akin to option discovery (Sutton et al.,
1999). WhileREGALhas a similar hierarchy, it differs in
that its skills are symbolic, interpretable, and editable, as
opposed to HRL policies, which typically are not.
Limitations. As mentioned in connection to HRL, the
functionsREGALlearns are code-based. This can make
them less flexible than functions parameterized by neural
networks (e.g., Andreas et al., 2016), especially in domains
where the environment can change dynamically, e.g., naviga-
tion tasks. However,REGAL’s verification-based pruning
means that no functions would be discovered in these cases.
Relatedly, not every domain has reusable abstractions, and
not every example stands to benefit from them; the primi-
tives for a domain may already be suitably abstract, e.g., if
they already form a DSL. Finally, in Appendix B.1 we see
thatREGAL’s abstractions are not necessarily the same as
those a human would choose.
```
## 8. Conclusion.

```
We introduceREGAL, a gradient-free approach to learning
abstractions from a small set of examples. Our experimental
results show that abstractions fromREGALimprove the
accuracy of programs predicted by a variety of LLMs across
five diverse domains. Furthermore,REGALabstractions
are reusable and general, allowing them to be applied across
examples for a given task. In our analysis, we find that
the functions learned byREGALcodify commonly-used
subroutines as well as task dynamics. Our error analysis
indicates thatREGAL’s improvements come both from
function reuse as well as simplification of the reasoning
involved in program prediction.
```

## Acknowledgements

We thank Yichen Jiang, Justin Chen, Jaehong Yoon, and
Swarnadeep Saha, as well as the anonymous reviewers, for
their valuable feedback on the paper. This work was sup-
ported by NSF-AI Engage Institute DRL-2112635, DARPA
Machine Commonsense (MCS) Grant N66001-19-2-4031,
and the Accelerate Foundation Models Research program.
The views contained in this article are those of the authors
and not of the funding agencies.

## Impact Statement

Our work aims to learn symbolic functions given a set of
demonstrations; this has the potential to improve LLM pre-
dictions not only in terms of accuracy but also in terms of
interpretability and trustworthiness. Unlike the mechanisms
of an LLM itself, a Python function is natively interpretable
by a human and can be debugged. Furthermore, results ob-
tained by executing such a function are inherently faithful,
in that we can identify the exact trace of operations that gen-
erated the result (Lyu et al., 2023). Our work does not have
more potential for negative use than typical LLM-based sys-
tems and is subject to the biases inherent to these models and
the datasets they are trained on (Weidinger et al., 2021). As
with any system generating code, particular caution should
be taken before executing snippets with the potential to
damage the execution environment (Ruan et al., 2023).

## References

Abelson, H. and DiSessa, A.Turtle geometry: The computer
as a medium for exploring mathematics. MIT press, 1986.

Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O.,
David, B., Finn, C., Fu, C., Gopalakrishnan, K., Hausman,
K., Herzog, A., Ho, D., Hsu, J., Ibarz, J., Ichter, B.,
Irpan, A., Jang, E., Ruano, R. J., Jeffrey, K., Jesmonth,
S., Joshi, N., Julian, R., Kalashnikov, D., Kuang, Y., Lee,
K.-H., Levine, S., Lu, Y., Luu, L., Parada, C., Pastor,
P., Quiambao, J., Rao, K., Rettinghouse, J., Reyes, D.,
Sermanet, P., Sievers, N., Tan, C., Toshev, A., Vanhoucke,
V., Xia, F., Xiao, T., Xu, P., Xu, S., Yan, M., and Zeng,
A. Do as i can and not as i say: Grounding language in
robotic affordances. InarXiv preprint arXiv:2204.01691,
2022.

Andreas, J., Rohrbach, M., Darrell, T., and Klein, D. Neural
module networks. InProceedings of the IEEE conference
on computer vision and pattern recognition, pp. 39–48,
2016.

Barto, A. G. and Mahadevan, S. Recent advances in hier-
archical reinforcement learning.Discrete event dynamic
systems, 13(1-2):41–77, 2003.

```
Bengio, Y., Louradour, J., Collobert, R., and Weston, J.
Curriculum learning. InProceedings of the 26th annual
international conference on machine learning, pp. 41–48,
2009.
```
```
Bogin, B., Gupta, S., Clark, P., and Sabharwal, A. Lever-
aging code to improve in-context learning for semantic
parsing.arXiv preprint arXiv:2311.09519, 2023.
```
```
Bowers, M., Olausson, T. X., Wong, L., Grand, G., Tenen-
baum, J. B., Ellis, K., and Solar-Lezama, A. Top-down
synthesis for library learning.Proceedings of the ACM on
Programming Languages, 7(POPL):1182–1213, 2023.
```
```
Cai, T., Wang, X., Ma, T., Chen, X., and Zhou, D.
Large language models as tool makers. arXiv preprint
arXiv:2305.17126, 2023.
```
```
Chen, W., Ma, X., Wang, X., and Cohen, W. W. Program
of thoughts prompting: Disentangling computation from
reasoning for numerical reasoning tasks.arXiv preprint
arXiv:2211.12588, 2022.
```
```
Cho, J., Zala, A., and Bansal, M. Visual programming for
text-to-image generation and evaluation.Thirty-seventh
Conference on Neural Information Processing Systems
(NeurIPS), 2023.
```
```
Cotˆ ́e, M.-A., K ́adar, A., Yuan, X., Kybartas, B., Barnes, T., ́
Fine, E., Moore, J., Hausknecht, M., El Asri, L., Adada,
M., et al. Textworld: A learning environment for text-
based games. InComputer Games: 7th Workshop, CGW
2018, Held in Conjunction with the 27th International
Conference on Artificial Intelligence, IJCAI 2018, Stock-
holm, Sweden, July 13, 2018, Revised Selected Papers 7,
pp. 41–75. Springer, 2019.
```
```
Downey, A.Think python. ” O’Reilly Media, Inc.”, 2012.
```
```
Dziri, N., Lu, X., Sclar, M., Li, X. L., Jian, L., Lin, B. Y.,
West, P., Bhagavatula, C., Bras, R. L., Hwang, J. D., et al.
Faith and fate: Limits of transformers on compositionality.
arXiv preprint arXiv:2305.18654, 2023.
```
```
Ellis, K., Wong, L., Nye, M., Sable-Meyer, M., Morales, L., ́
Hewitt, L., Cary, L., Solar-Lezama, A., and Tenenbaum,
J. B. Dreamcoder: Bootstrapping inductive program syn-
thesis with wake-sleep library learning. InProceedings
of the 42nd ACM SIGPLAN International Conference on
Programming Language Design and Implementation, pp.
835–850, 2021.
```
```
Flynn, B. B. and Jacobs, F. R. Applications and implemen-
tation: an experimental comparison of cellular (group
technology) layout with process layout. Decision Sci-
ences, 18(4):562–581, 1987.
```

Furnkranz, J., Gamberger, D., and Lavra ̈ c, N.ˇ Foundations
of rule learning. Springer Science & Business Media,
2012.

Futuyma, D. J. and Moreno, G. The evolution of ecological
specialization.Annual review of Ecology and Systematics,
19(1):207–233, 1988.

Grand, G., Wong, L., Bowers, M., Olausson, T. X., Liu,
M., Tenenbaum, J. B., and Andreas, J. Learning in-
terpretable libraries by compressing and documenting
code. InIntrinsically-Motivated and Open-Ended Learn-
ing Workshop@ NeurIPS2023, 2023.

Gupta, S., Shah, R., Mohit, M., Kumar, A., and Lewis,
M. Semantic parsing for task oriented dialog using hi-
erarchical representations. InProceedings of the 2018
Conference on Empirical Methods in Natural Language
Processing, pp. 2787–2792, 2018.

Gupta, T. and Kembhavi, A. Visual programming: Compo-
sitional visual reasoning without training. InProceedings
of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pp. 14953–14962, 2023.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart,
S., Tang, E., Song, D., and Steinhardt, J. Measuring
mathematical problem solving with the math dataset. In
Thirty-fifth Conference on Neural Information Process-
ing Systems Datasets and Benchmarks Track (Round 2),
2021.

Huang, W., Abbeel, P., Pathak, D., and Mordatch, I. Lan-
guage models as zero-shot planners: Extracting action-
able knowledge for embodied agents. arXiv preprint
arXiv:2201.07207, 2022.

Huang, W., Wang, C., Zhang, R., Li, Y., Wu, J., and Fei-Fei,
L. Voxposer: Composable 3D value maps for robotic
manipulation with language models. InConference on
Robot Learning, pp. 540–562. PMLR, 2023.

Khot, T., Trivedi, H., Finlayson, M., Fu, Y., Richardson, K.,
Clark, P., and Sabharwal, A. Decomposed prompting:
A modular approach for solving complex tasks. InThe
Eleventh International Conference on Learning Repre-
sentations, 2023.

Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B.
Human-level concept learning through probabilistic pro-
gram induction.Science, 350(6266):1332–1338, 2015.

Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., and
Chen, W. What makes good in-context examples for
GPT-3? In Agirre, E., Apidianaki, M., and Vulic, I. ́
(eds.),Proceedings of Deep Learning Inside Out (DeeLIO
2022): The 3rd Workshop on Knowledge Extraction and

```
Integration for Deep Learning Architectures, pp. 100–
114, Dublin, Ireland and Online, May 2022. Association
for Computational Linguistics.
```
```
Liu, X., Yu, H., Zhang, H., Xu, Y., Lei, X., Lai, H., Gu, Y.,
Ding, H., Men, K., Yang, K., et al. AgentBench: Evaluat-
ing LLMs as agents.arXiv preprint arXiv:2308.03688,
2023.
```
```
Lu, P., Qiu, L., Chang, K.-W., Wu, Y. N., Zhu, S.-C., Ra-
jpurohit, T., Clark, P., and Kalyan, A. Dynamic prompt
learning via policy gradient for semi-structured mathemat-
ical reasoning. InThe Eleventh International Conference
on Learning Representations, 2022.
```
```
Lu, P., Peng, B., Cheng, H., Galley, M., Chang, K.-W.,
Wu, Y. N., Zhu, S.-C., and Gao, J. Chameleon: Plug-and-
play compositional reasoning with large language models.
arXiv preprint arXiv:2304.09842, 2023.
```
```
Lyu, Q., Havaldar, S., Stein, A., Zhang, L., Rao, D.,
Wong, E., Apidianaki, M., and Callison-Burch, C.
Faithful Chain-of-Thought Reasoning. arXiv preprint
arXiv:2301.13379, 2023.
```
```
Majumder, B. P., Mishra, B. D., Jansen, P., Tafjord, O.,
Tandon, N., Zhang, L., Callison-Burch, C., and Clark,
P. Clin: A continually learning language agent for
rapid task adaptation and generalization.arXiv preprint
arXiv:2310.10134, 2023.
```
```
Marcus, G. F., Vijayan, S., Bandi Rao, S., and Vishton, P. M.
Rule learning by seven-month-old infants.Science, 283
(5398):77–80, 1999.
```
```
McConnell, S.Code complete. Pearson Education, 2004.
```
```
Mialon, G., Dess`ı, R., Lomeli, M., Nalmpantis, C., Pa-
sunuru, R., Raileanu, R., Roziere, B., Schick, T., Dwivedi-`
Yu, J., Celikyilmaz, A., et al. Augmented Language Mod-
els: a Survey.arXiv preprint arXiv:2302.07842, 2023.
```
```
O’Donnell, T. J. Productivity and reuse in language: A
theory of linguistic computation and storage. MIT Press,
2015.
```
```
OpenAI. New and improved embedding model,
```
2022. URL https://openai.com/blog/
new-and-improved-embedding-model.

```
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V.,
Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P.,
Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cour-
napeau, D., Brucher, M., Perrot, M., and Duchesnay, E.
Scikit-learn: Machine learning in Python. Journal of
Machine Learning Research, 12:2825–2830, 2011.
```

Prasad, A., Koller, A., Hartmann, M., Clark, P., Sabharwal,
A., Bansal, M., and Khot, T. Adapt: As-needed decompo-
sition and planning with language models.arXiv preprint
arXiv:2311.05772, 2023.

Qian, C., Han, C., Fung, Y., Qin, Y., Liu, Z., and Ji, H.
Creator: Tool creation for disentangling abstract and con-
crete reasoning of large language models. InFindings of
the Association for Computational Linguistics: EMNLP
2023 , pp. 6922–6939, 2023.

Qin, Y., Liang, S., Ye, Y., Zhu, K., Yan, L., Lu, Y., Lin, Y.,
Cong, X., Tang, X., Qian, B., et al. ToolLLM: Facilitating
large language models to master 16000+ real-world APIs.
arXiv preprint arXiv:2307.16789, 2023.

Roy, S., Thomson, S., Chen, T., Shin, R., Pauls, A., Eisner,
J., and Van Durme, B. Benchclamp: A benchmark for
evaluating language models on semantic parsing.arXiv
preprint arXiv:2206.10668, 2022.

Roziere, B., Gehring, J., Gloeckle, F., Sootla, S., Gat, I.,
Tan, X. E., Adi, Y., Liu, J., Remez, T., Rapin, J., et al.
Code llama: Open foundation models for code. arXiv
preprint arXiv:2308.12950, 2023.

Ruan, Y., Dong, H., Wang, A., Pitis, S., Zhou, Y., Ba, J.,
Dubois, Y., Maddison, C., and Hashimoto, T. Identifying
the risks of LM agents with an LM-emulated sandbox. In
NeurIPS 2023 Foundation Models for Decision Making
Workshop, 2023.

Saha, S., Zhang, S., Hase, P., and Bansal, M. Summariza-
tion programs: Interpretable abstractive summarization
with neural modular trees. InThe Eleventh International
Conference on Learning Representations, 2022.

Schick, T., Dwivedi-Yu, J., Dess`ı, R., Raileanu, R., Lomeli,
M., Zettlemoyer, L., Cancedda, N., and Scialom, T. Tool-
former: Language models can teach themselves to use
tools.arXiv preprint arXiv:2302.04761, 2023.

Shin, R. and Van Durme, B. Few-shot semantic parsing
with language models trained on code. InProceedings of
the 2022 Conference of the North American Chapter of
the Association for Computational Linguistics: Human
Language Technologies, pp. 5417–5425, 2022.

Singh, I., Blukis, V., Mousavian, A., Goyal, A., Xu, D.,
Tremblay, J., Fox, D., Thomason, J., and Garg, A. Prog-
Prompt: Generating situated robot task plans using Large
Language Models. In2023 IEEE International Confer-
ence on Robotics and Automation (ICRA), pp. 11523–

11530. IEEE, 2023.

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid,
A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A.,
Garriga-Alonso, A., et al. Beyond the imitation game:

```
Quantifying and extrapolating the capabilities of language
models.arXiv preprint arXiv:2206.04615, 2022.
```
```
Sur ́ıs, D., Menon, S., and Vondrick, C. ViperGPT: Vi-
sual inference via Python execution for reasoning.arXiv
preprint arXiv:2303.08128, 2023.
```
```
Sutton, R. S., Precup, D., and Singh, S. Between MDPs and
semi-MDPs: A framework for temporal abstraction in
reinforcement learning.Artificial intelligence, 112(1-2):
181–211, 1999.
```
```
Suzgun, M., Scales, N., Scharli, N., Gehrmann, S., Tay, Y., ̈
Chung, H. W., Chowdhery, A., Le, Q. V., Chi, E. H.,
Zhou, D., et al. Challenging Big-Bench tasks and
whether chain-of-thought can solve them.arXiv preprint
arXiv:2210.09261, 2022.
```
```
Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu,
Y., Fan, L., and Anandkumar, A. Voyager: An open-
ended embodied agent with large language models.arXiv
preprint arXiv:2305.16291, 2023.
```
```
Wang, Z., Fried, D., and Neubig, G. Trove: Inducing veri-
fiable and efficient toolboxes for solving programmatic
tasks.arXiv preprint arXiv:2401.12869, 2024.
```
```
Ward Jr, J. H. Hierarchical grouping to optimize an objective
function.Journal of the American statistical association,
58(301):236–244, 1963.
```
```
Weidinger, L., Mellor, J., Rauh, M., Griffin, C., Uesato,
J., Huang, P.-S., Cheng, M., Glaese, M., Balle, B.,
Kasirzadeh, A., et al. Ethical and social risks of harm
from language models.arXiv preprint arXiv:2112.04359,
2021.
```
```
Winograd, T. Understanding natural language.Cognitive
psychology, 3(1):1–191, 1972.
```
```
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C.,
Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M.,
Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite,
Y., Plu, J., Xu, C., Scao, T. L., Gugger, S., Drame, M.,
Lhoest, Q., and Rush, A. M. Transformers: State-of-
the-art natural language processing. InProceedings of
the 2020 Conference on Empirical Methods in Natural
Language Processing: System Demonstrations, pp. 38–
45, Online, October 2020. Association for Computational
Linguistics.
```
```
Wong, L., Ellis, K. M., Tenenbaum, J., and Andreas, J.
Leveraging language to learn program abstractions and
search heuristics. InInternational Conference on Ma-
chine Learning, pp. 11193–11204. PMLR, 2021.
```
```
Wong, L., Mao, J., Sharma, P., Siegel, Z. S., Feng, J., Ko-
rneev, N., Tenenbaum, J. B., and Andreas, J. Learning
```

```
adaptive planning representations with natural language
guidance.arXiv preprint arXiv:2312.08566, 2023.
```
Yang, C.The price of linguistic productivity: How children
learn to break the rules of language. MIT press, 2016.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K. R., and Cao, Y. React: Synergizing reasoning and
acting in language models. InThe Eleventh International
Conference on Learning Representations, 2023.

Yasunaga, M., Chen, X., Li, Y., Pasupat, P., Leskovec, J.,
Liang, P., Chi, E. H., and Zhou, D. Large language models
as analogical reasoners.arXiv preprint arXiv:2310.01714,
2023.

Yuan, L., Chen, Y., Wang, X., Fung, Y. R., Peng, H.,
and Ji, H. Craft: Customizing LLMs by creating and
retrieving from specialized toolsets. arXiv preprint
arXiv:2309.17428, 2023.

Zelle, J. M. and Mooney, R. J. Learning to parse database
queries using inductive logic programming. InProceed-
ings of the national conference on artificial intelligence,
pp. 1050–1055, 1996.

## Appendix

## A. Methods

A.1. Preprocessing

Adding comments. To add comments, we first use a zero-
shot prompt to break the query down into its constituent
parts; for example, a LOGO query like“Place 4 small semi-
circles in a row”is broken down into“1. place semicircles

2. small semicircles 3. in a row 4. 4 semicircles. We then
include this decomposition in a prompt asking the model
to add comments to the code. After adding comments, we
verify the code first with exact match (excluding comment
strings) and then use execution accuracy if exact match fails.

A.2. Training

Code Bank Editing. Our Code Bank editing prompt asks
the model to produce a CoT-style output, first specifying
whythe failing unit tests failed and then proposing an edit
for the function. We then execute the stored demonstrations
for that function with the new version; if there are more
passing cases after refactoring, we replace the function. If
the new function’s signature differs from the old, we use a
simple prompt to refactor the unit tests to accommodate the
new function.

Code Bank Pruning. For each function, given a set of
passing programsPand failing programsF, we compute

```
Table 5.Dataset statistics. We list the number of primitive opera-
tions in the programs (aside from built-in Python functions).
```
```
Dataset Train Dev Test # Primitives
LOGO 200 100 111 9
Date 66 113 180 4
TextCraft 190 50 77 3
MATH (Alg.) 194 61 74 0
TabMWP 194 60 74 0
```
```
a scores=|P|−
```
### P

```
p∈F^1 /np, wherenpis the number of
functions used inp. In other words, the function receives+
for each passing program it participates in, and a negative
score inversely proportional to the number of functions in
the program (since naively, the failure could be attributed
to any of the functions). Functions are pruned if they have
been used a sufficient number of times andsfalls below a
thresholdθ(set to 0 for all experiments).
```
```
A.3. Testing
```
```
In our test-time agent, we use ChromaDB^6 for indexing and
retrieval with OpenAI Ada embeddings. ICL examples are
retrieved from the training data and from the Demo Bank
using query similarity. We limit the number of Code Bank
functions to 20, using the similarity between the query and
the function name for retrieval. The Code Bank is pruned
once before testing.
```
```
A.4. Models
For GPT-3.5, we use thegpt- 3. 5 - turboversion (0613). All
CodeLlama models use theCodeLlama-∗-Instruct-hf
versions, and we use thelemur-70b-v1version of Lemur.
For the latter two open-source models, we use the check-
points from HuggingFace (Wolf et al., 2020).
```
```
A.5. Data
```
```
A.5.1. LOGO
```
```
LOGO data is generated from a synchronous text-code gram-
mar, and pairs procedurally-generated language commands
like“three small triangles in a row”with a correspond-
ing Turtle graphics program; however, the original LOGO
dataset is expressed in a Lisp-style functional syntax. While
this facilitates the application of helpful data structures for
efficient code search (Ellis et al., 2021; Bowers et al., 2023),
object-oriented languages like Python are far more common
in practice. As a result, they are represented more in LLM
pretraining data, which has been shown to contribute to pars-
ing performance (Bogin et al., 2023). To account for this,
we write an AST-based parsing script to translate the LOGO
```
(^6) https://github.com/chroma-core/chroma/


Algorithm 1REGAL: Training Algorithm

```
Input:X= (q,p)// Training data: (query, program)
Params:editEvery, pruneEvery, thresholdθ
Output:CodeBankC, DemoBankD
C←∅,D←∅// Initialization, i.e., no refactoring
// Preprocessing data via clustering and sorting by difficulty
P ←preprocessAndGroupData(X)
for indexg, batchG ∈Pdo
// Refactor programs in groupGbased on the current Code-
BankC. Returns new programs and helper functions.
(pnew 1 ,h 1 ),...,(pnewk ,hk)=refactorBatch(G,C)
Hnew←{h 1 ,···,hk}// Set of new helper functionsH
// Verifying that the gold program and the refactored pro-
gram yield the same result when executed via indicator
δnew.
δnew1:k ←verify(H,C,{pnewi }ki=1,{pi}ki=1)
for i∈{i:δnewi =False}do
(pretryi ,hretryi )←retry(pi,pnewi ,C)
δnewi ←verify(hretryi ∪H,C,pnew,p)
ifδinew=Truethen// Update if retry succeeds
pnewi ←pretryi
Hnew[i]←hretryi
// update CodeBankCwith successful helper functions
C←C+Hnew[i]fori∈{i:δnewi =True}
// update DemoBankDwith refactored programs
for i∈{ 1 ,...,k}do
D←D+ (pnewi ,δnewi )
// edit and prune CodeBank
ifg(mod editEvery) = 0then
C←editCodeBank(C,D)
ifg(mod pruneEvery) = 0then
C,D←pruneCodeBank(C,D,θ)
returnC,D
```
dataset into Python. The LOGO dataset was released under
an MIT License.

Primitives. Table 6 describes the primitives available in
the LOGO library. Note that these are in addition to all
Python primitives. We also provide all agents with several
hard-coded values for long loops and small steps so that
they can draw round shapes.HALFINFis the number of
steps required to draw a semicircle. EPSDISTis a small
distance, andEPSANGLEis a small angle.

```
A.5.2. DATE UNDERSTANDING
```
Date understanding involves both mathematical reasoning
and parsing. Each question poses a word problem that re-
quires reasoning about dates and times. For example, prob-
lems ask questions like:“On May 9th, 2017 Jane bought
40 eggs. She ate one per day. Today she ran out of eggs.
What is the date 10 days ago in MM/DD/YYYY?”. These
types of questions are especially hard for LLMs to answer

```
Algorithm 2REGAL: Testing Algorithm
Input:Q,C,D,X// Test queriesQ, Code BankC, Demo
BankD, Training dataX=(query, program)
Params:ICL BudgetM, ICL Percentager
Output:Predicted programsPˆ
Mdemo←r∗M
Mtrain←M−Mdemo
Pˆ←∅
for q∈Xdo
H←retrieve(q,C,20)// retrieve up to 20 helper
functions conditioned on the query
Xdemo←retrieve(q,D,Mdemo)// retrieve helper
demos fromD
Xtrain←retrieve(q,X,Mtrain)// retrieve primi-
tive demos fromX
I←createPrompt(H,Xdemo,Xtrain)
pˆ←LLM(I)// generate program
Pˆ←Pˆ+ ˆp
returnPˆ
```
```
directly. Lyu et al. (2023) approached this task as a pro-
gram prediction task, wherein an LLM predicts a Python
script that gives the answer to the question when executed.
This paradigm is especially helpful for Date, as there are
existing Python libraries that can perform math on dates,
such asdatetime() anddateutil(). While predicting pro-
grams with these libraries results in strong performance
as compared to LLM-based reasoning, Lyu et al. (2023)
method predicts programs one at a time, leaving the benefits
of shared sub-routines on the table. We use the programs
predicted by Lyu et al. (2023) as a starting point for our
refactoring process. Table 7 describes the Python libraries
called by Lyu et al. (2023)’s programs, which we treat as
the primitives for Date. Date was released under an Apache
2.0 License.
```
```
A.5.3. TEXTCRAFT
TextCraft consists of goal queries paired with crafting
recipes. Recipes are presented with distractors, making
the parsing task challenging. Furthermore, the agent must
reason about preconditions, as items can only be crafted
when the requisite ingredients have been collected.
The queries ask for a particular item to be crafted. For
example the query can be “craft behive” along with crafting
commands:
```
```
craft 4 oak planks using 1 oak log
craft 1 honeycomb block using 4 honeycomb
craft 1 beehive using 6 planks and 3 honeycombs
craft 1 white bed using 3 white wool, 3 planks,
etc.
```

```
Table 6.LOGO Primitives
Primitive Description
forward(dist) move forwarddistunits
left(theta) rotate left bythetade-
grees
right(theta) rotate right bythetade-
grees
penup() lift the pen (stop draw-
ing)
pendown() put the pen down (start
drawing)
teleport(x,y,theta) move to position(x,y)
with angletheta
heading() get the current angle of
the turtle
isdown() check if the pen is down
embed(program,vars) runs the code inprogram
using the current context
varsand teleports back
to the original position.
```
```
Table 7.Date Primitives
Primitive Description
date() returns adateobject
time() returns atimeobject
relativedelta(time) performs addition/sub-
traction oftime, which
can be days, weeks,
months, or years.
strftime(format) prints the date in the spec-
ified format
```
The environment has three primitives:inventory,craft,
andgetwhich we convert into Python variants (Table 8).
This dataset uses the Apache 2.0 license. To obtain primitive
programs in the train set, we use all the depth-2 instances not
in the test set, and use theADAPT(Prasad et al., 2023) tra-
jectories withd= 4. We then perform rule-based translation
of environment commands into calls to primitive operations
and filter out erroneous actions (based on the saved textual
environment feedback) and all thought statements.

```
A.5.4. MATH
```
MATH (Hendrycks et al., 2021) contains challenging math
word problems with open-ended answers (i.e. not multiple-
choice). Unlike LOGO, Date, and TextCraft, MATH does
not have any primitives, as each problem can be solved using
standard Python math tools (addition, subtraction, etc.); this
makes MATH a challenging setting for discovering helper
functions, as the space of programs is less constrained.

```
Table 8.TextCraft Primitives
Primitive Description
getObject(objname) getobjnamefrom the
environment
craftObject(objname,
[ingredients])
```
```
craftobjnameusing the
list of ingredients
checkInventory() return the contents of the
inventory
```
### A.5.5. TABMWP

```
Like MATH, TabMWP (Lu et al., 2022) is a dataset of
word problems. In this case, the word problems pertain to
tabular data, where each problem asks questions involving
computation over tabular data. This dataset has been used
in prior work on tool learning, e.g. Yuan et al. (2023). Like
MATH, there is no existing domain-specific language for
TabMWP.
```
## B. Analysis

```
B.1. What kinds of programs are discovered
Figs. 5 to 7 show examples of the discovered programs most
commonly used by the agent.
```
```
def drawsmall5gon():
for i in range(5):
forward(2)
left(72.0)
def drawsemicircle():
for i in range(HALFINF):
forward(EPSDIST * 2):
left(EPSANGLE)
```
```
Figure 5.Examples of discovered programs for LOGO as men-
tioned in Figs. 1 and 3. As the name suggests,drawsmall5gon()
draws a small-size pentagon anddrawsemicircle() draws a
small semicircle.
```
```
def getdatetoday(dateobj):
return dateobj
def getdateoneweekfromtoday(dateobj):
return dateobj + relativedelta(weeks=1)
def getdateoneweekago(dateobj):
return dateobj −relativedelta(weeks=1)
def getdateoneyearago(datetoday):
return datetoday− relativedetla(years
=1)
```
```
Figure 6.Examples of common discovered programs for Date as
mentioned in Fig. 3. All the helpers functions and variables are
named intuitively reflection their functionality.
```

```
def craftobjectwithingredients(target,
ingredients):
inventory = checkinventory()
for ingredient iningredients:
if ingredientnot in inventory:
getobjectfromenv(ingredient)
craftobject(target, ingredients)
def checkandgetobject(target):
inventory = checkinventory()
if targetnot in inventory:
getobject(target)
```
Figure 7.Examples of common discovered programs for TextCraft
as shown in Figs. 1 and 2.craftobjectwithingredients()
encapsulate the game dynamics of TextCraft as it first fetches the in-
ventory, and every ingredient not in the inventory is obtained from
the environment prior to using the crafting the target object. Simi-
larly, the helpercheckandgetobject() gets an object from the
environment if it is not already in the inventory.

```
def calculatemidpoint(point1, point2):
x = (point1[0] + point2[0]) / 2
y = (point1[1] + point2[1]) / 2
return(x, y)
def calculatesquare(num):
return num * * 2
```
Figure 8.Examples of discovered programs for MATH.
calculatemidpointis an example of reuse, where RE-
GALfinds a frequently-used functionality and encapsulates
it, while incalculatesquareREGALwraps a fairly easy
function with a more informative name.

## C. Hyperparameters

Table 9 lists the refactoring and testing hyperparameters
used for each domain.

C.1. Varying Batch Size

Fig. 10 shows the performance of a CodeLlama-13B agent
on the Date dev set forREGALabstractions trained using
different batch sizes. We see that performance can vary
substantially and non-linearly with batch size, with 3 being
the best setting and 5 a close second.

```
Table 9.Hyperparameter settings for all experiments
Setting LOGO Date TextCraft
Rounds of refactoring 3 1 1
editEvery 5 5 5
pruneEvery 5 5 5
Add comments True False False
Batch size 5 3 4
Filtering threshold 0.0 0.0 0.
Filter before testing True True False
ICL budget ratio 0.5 0.5 0.
```
```
def calculatetotalcost(packages, items):
totalcost = 0
for item initems:
totalcost += packages[item]
return totalcost
def addprices(prices):
totalcost = 0
for price inprices.values():
totalcost += price
return totalcost
```
```
Figure 9.Examples of discovered programs for TabMWP, reflect-
ing the domain, which often involves costs and prices.
```
```
2 3 4 5 6
Batch size
```
```
0.
```
```
0.
```
```
Accuracy
```
```
Figure 10.Validation performance of CodeLlama-13B withRE-
GAL abstractions trained using different batch sizes.
```
## D. Prompts

```
Below, we detail the prompts used in all sections.
```

Table 10.Batch refactoring prompt (refactorBatch()). Comments indicate where text is repeated. Brackets indicate variables filled in
by the environment. Note that “<>strings” are passed as-is to the LLM.

```
P l e a s e r e w r i t e t h e f o l l o w i n g t w o p r o g r a m s t o b e more e f f i c i e n t.
{p r i m i t i v e d e s c r i p t i o n s t r i n g}
The r e s u l t i n g p r o g r a m s MUST e x e c u t e t o t h e same r e s u l t a s t h e o r i g i n a l p r o g r a m s.
S t a r t by w r i t i n g h e l p e r f u n c t i o n s t h a t c a n r e d u c e t h e s i z e o f t h e c o d e.
You c a n a l s o c h o o s e f r o m t h e f o l l o w i n g h e l p e r f u n c t i o n s :
{c o d e b a n k f u n c t i o n d e f i n i t i o n s}
```
```
/ / r e p e a t e d f o r a l l i n b a t c h
QUERY{i}: {q u e r y}
PROGRAM{i}: {p r o g r a m}
```
```
P l e a s e f o r m a t y o u r a n s w e r a s :
/ / r e p e a t e d f o r i
NEW PROGRAM{i}:
/ / o n c e a t e n d
NEW HELPERS :
```
```
Do n o t i n c l u d e a n y t e x t t h a t i s n o t v a l i d P y t h o n c o d e.
R e c a l l t h a t no m a t t e r w h a t , y o u r p r o g r a m MUST b e f o r m a t t e d i n t h e f o l l o w i n g f a s h i o n :
/ / r e p e a t e d f o r i
NEW PROGRAM{i}:
# T h o u g h t s :
# 1. The q u e r y a s k s f o r : <q u e r y i n t e n t i o n>
2 .<q u e r y> c a n b e s o l v e d by<c o m p o n e n t s>.
# 3. I w i l l u s e h e l p e r f u n c t i o n<f u n c t i o n> t o <g o a l>.
<c o d e f o r p r o g r a m {i}>
```
```
T r y t o make y o u r new p r o g r a m s a s s h o r t a s p o s s i b l e by i n t r o d u c i n g s h a r e d h e l p e r f u n c t i o n s.
H e l p e r f u n c t i o n p a r a m e t e r s s h o u l d b e a s g e n e r a l a s p o s s i b l e a n d h e l p e r f u n c t i o n s
s h o u l d b e i n f o r m a t i v e l y named.
{l o g o s p e c i a l i n s t r}
```
```
logospecialinstructions=
I f t h e o r i g i n a l f u n c t i o n u s e s ‘ embed ‘ , y o u w i l l l i k e l y n e e d t o u s e ‘ embed ‘ i n y o u r v e r s i o n
```
. A l l c o d e t o b e r e p e a t e d n e e d s t o b e i n c l u d e d w i t h i n t h e t r i p l e q u o t e s p a s s e d t o
embed.

```
Table 11.Query decomposition prompt. Output is used by the comment prompt in Table 12
```
```
You a r e a n e x p e r t c o d e r. F o r e a c h q u e r y b e l o w , d e c o m p o s e i t i n t o i t s p a r t s.
E x a m p l e :
Q u e r y : Do some a c t i o n 5 t i m e s a n d t h e n do a n o t h e r a c t i o n
Q u e r y ( d e c o m p o s e d ) :
The q u e r y a s k s : Do some a c t i o n a n d t h e n do a n o t h e r a c t i o n
T h i s c a n b e d e c o m p o s e d i n t o :
1. r e p e a t a n a c t i o n
2. some a c t i o n
3. a n o t h e r a c t i o n
```
```
Q u e r y : {q u e r y}
Q u e r y ( d e c o m p o s e d ) :
```

```
Table 12.Prompt to add comments to primitive programs. Takes output of Table 11 as input.
```
P l e a s e a d d c o m m e n t s t o t h e f o l l o w i n g p r o g r a m t o e x p l a i n w h a t e a c h c h u n k o f c o d e d o e s w i t h
r e s p e c t t o t h e q u e r y.
F i r s t , d e c o m p o s e t h e q u e r y i n t o p a r t s. Then comment t h e c o d e w i t h t h e q u e r y p a r t s.
E x a m p l e :
Q u e r y : Do some a c t i o n a n d t h e n do a n o t h e r a c t i o n
Code :
d o s o m ea c t i o n ( )
d o a n o t h e r a c t i o n ( )

Q u e r y : Do some a c t i o n 5 t i m e s a n d t h e n do a n o t h e r a c t i o n
Q u e r y ( d e c o m p o s e d ) :
The q u e r y a s k s : Do some a c t i o n a n d t h e n do a n o t h e r a c t i o n
T h i s c a n b e d e c o m p o s e d i n t o :
1. r e p e a t a n a c t i o n
2. some a c t i o n
3. a n o t h e r a c t i o n
Commented c o d e :
# r e p e a t a n a c t i o n
f o r i i n r a n g e ( 5 ) :
# do some a c t i o n
d o s o m e a c t i o n ( )
# do a n o t h e r a c t i o n
d o a n o t h e r a c t i o n ( )

{p r i m i t i v e d e s c r i p t i o n}

Q u e r y : {q u e r y}
Code :
{p r o g r a m}

Q u e r y ( d e c o m p o s e d ) :
{o u t p u t f r o m d e c o m p o s e p r o m p t}


```
Table 13.Prompt foreditCodeBank.
```
```
R e f a c t o r t h e f o l l o w i n g f u n c t i o n t o i m p r o v e p e r f o r m a n c e.
FUNCTION :
‘ ‘ ‘
{f u n c s t r}
‘ ‘ ‘
```
```
{l i b r a r y s t r}
```
```
You may a l s o u s e t h e f o l l o w i n g h e l p e r f u n c t i o n s :
{c o d e b a n k s t r}
```
```
T r y t o i n c r e a s e t h e n u m b e r o f p a s s i n g p r o g r a m s. T r y t o make p r o g r a m s g e n e r a l. F o r e x a m p l e ,
y o u c a n a d d p a r a m e t e r s i n s t e a d o f h a r d c o d e d v a l u e s o r c a l l o t h e r h e l p e r f u n c t i o n s.
F i r s t , f o r e a c h f a i l i n g q u e r y , e x p l a i n why t h e p r o g r a m s do n o t a c c o m p l i s h t h e q u e r y ’ s
g o a l. O u t p u t t h i s r e a s o n i n g a s :
T h o u g h t s :
1. The f u n c t i o n p a s s e s some t e s t s a n d f a i l s o t h e r s b e c a u s e<r e a s o n>.
2. The f a i l i n g q u e r i e s<r e p e a t q u e r i e s h e r e> a s k e d f o r<i n t e n t>.
3. The p r o g r a m f a i l e d b e c a u s e<r e a s o n>.
4. T h i s c a n b e a d d r e s s e d by<c h a n g e>.
Then o u t p u t y o u r p r o g r a m s o t h a t a l l t e s t c a s e s p a s s , u s i n g t h e f o l l o w i n g f o r m a t : NEW
PROGRAM: <p r o g r a m>
C u r r e n t l y , {f u n c. n a m e} p a s s e s i n {p a s s p e r c * 1 0 0 :. 1 f}\% o f c a s e s a n d f a i l s i n {f a i l p e r c
*1 0 0 :. 1 f}\%.
SUCCEEDED :
{e x a m p l e o f p a s s i n g c a s e}
FAILED :
{e x a m p l e o f f a i l i n g c a s e}
T h o u g h t s :
```
Table 14.Agent prompt for Python tasks like Date understanding. Note that the baseline andREGALagent use the same prompt, but
{codebankstr}is empty for the baseline agent, and the REGAL sees some demonstrations from the Demo Bank in{iclstring}.

```
Your t a s k i s t o s o l v e s i m p l e word p r o b l e m s by c r e a t i n g P y t h o n p r o g r a m s.
{c o d e b a n k s t r}
```
```
You w i l l b e g i v e n a q u e r y a n d h a v e t o p r o d u c e a p r o g r a m. {t h o u g h t s t r}
E x a m p l e s :
{i c l s t r i n g}
```
```
P l e a s e g e n e r a t e ONLY t h e c o d e t o p r o d u c e t h e a n s w e r a n d n o t h i n g e l s e.
Q u e r y : {q u e r y}
{t h o u g h t a n d}P r o g r a m :
```

```
Table 15.Prompt for the LOGO agent.
```
Your t a s k i s t o d r a w s i m p l e f i g u r e s u s i n g p y t h o n T u r t l e g r a p h i c s.
You w i l l u s e a c u s t o m t u r t l e l i b r a r y , s i m i l a r t o t h e b u i l t − i n l i b r a r y , w h i c h i s s u f f i c i e n t
f o r a l l t a s k s.

H e r e ’ s a d e s c r i p t i o n o f t h e c u s t o m l i b r a r y :
− f o r w a r d ( x ) : move f o r w a r d x p i x e l s
− l e f t ( t h e t a ) : r o t a t e l e f t by t h e t a d e g r e e s
− r i g h t ( t h e t a ) : r o t a t e r i g h t by t h e t a d e g r e e s
− p e n u p ( ) : s t o p d r a w i n g
− pendown ( ) : s t a r t d r a w i n g
− t e l e p o r t ( x , y , t h e t a ) : move t o p o s i t i o n ( x , y ) w i t h a n g l e t h e t a
− h e a d i n g ( ) : g e t t h e c u r r e n t a n g l e o f t h e t u r t l e
− i s d o w n ( ) : c h e c k i f t h e p e n i s down
− embed ( p r o g r a m , l o c a l v a r s ) : r u n s t h e c o d e i n p r o g r a m u s i n g t h e c u r r e n t c o n t e x t a n d
t e l e p o r t s b a c k t o t h e o r i g i n a l p o s i t i o n. A l l o w s y o u t o n e s t p r o g r a m s.
I m p l e m e n t a t i o n a l l y , embed g e t s t h e t u r t l e s t a t e ( i sd o w n , x , y , h e a d i n g ) , e x e c u t e s
p r o g r a m , t h e n r e t u r n s t o t h e o r i g i n a l s t a t e.
− s a v e ( p a t h ) : s a v e t h e p i c t u r e t o f i l e
{c o d e b a n k s t r}

You w i l l b e g i v e n a q u e r y a n d h a v e t o p r o d u c e a p r o g r a m. B e g i n y o u r p r o g r a m w i t h a comment
t h a t e x p l a i n s y o u r r e a s o n i n g. F o r e x a m p l e , y o u m i g h t w r i t e :\n # T h o u g h t : t h e q u e r y
a s k s f o r a l i n e , s o I w i l l u s e t h e f o r w a r d ( ) f u n c t i o n.
E x a m p l e s :
{i c l s t r i n g}

P l e a s e g e n e r a t e ONLY t h e c o d e t o p r o d u c e t h e a n s w e r a n d n o t h i n g e l s e.
Q u e r y : {q u e r y}
T h o u g h t a n d P r o g r a m :