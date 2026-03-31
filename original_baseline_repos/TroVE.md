## TROVE: Inducing Verifiable and Efficient Toolboxes

## for Solving Programmatic Tasks

```
Zhiruo Wang^1 Graham Neubig^1 Daniel Fried^1
```
### Abstract

```
Language models (LMs) can solve tasks such as
answering questions about tables or images by
writing programs. However, using primitive func-
tions often leads to verbose and error-prone pro-
grams, and higher-level functions require expert
design. To enable better solutions without hu-
man labor, we ask code LMs to curate reusable
high-level functions, and use them to write solu-
tions. We presentTROVE, atraining-freemethod
of inducing averifiable andefficient toolboxof
functions, by generating via using, growing, and
periodically trimming the toolbox. On 11 datasets
from math, table question answering, and image
reasoning tasks,TROVEconsistently yieldssim-
pler solutionswithhigher accuracythan baselines
usingCODELLAMAand previous methods us-
ingGPT, while using 79 - 98%smallertoolboxes.
TROVEfurther enables31%fasterand13%more
accurate human verificationthan baselines. With
the same pipeline, it createsdiverse functionsfor
varied tasks and datasets, providing insights into
their individual characteristics. Code and data are
available at https://github.com/zorazrw/trove.
```
### 1. Introduction

```
Generating code from natural language commands has long
been a method of choice for solving tasks such as question
answering (Zettlemoyer & Collins, 2007; Liang et al., 2011)
or agent navigation (Artzi & Zettlemoyer, 2013). Recently,
language models (LMs) have been used to write programs in
general-purpose languages such as Python, further expand-
ing code generation’s applicability (Yin & Neubig, 2017; Li
et al., 2022b; Cheng et al., 2023). These programs generally
rely on multiple function calls to Python built-in functions
or libraries such aspandas, as in the example in Figure 2.
However, in many cases relying on these primitive func-
tions can lead to programs that are tedious, complex, and
error-prone (Cai et al., 2023; Majumder et al., 2023). These
```
(^1) Language Technologies Institute, Carnegie Mellon University.
reusable
small functions^
toolbox
**Test**
questions
program
solutions
TroVE
(ours)

### x K

```
✔ training-free
✔ simple & accurate
✔ reusable tools
✔ easily verifiable
```
```
LM
```
**Test** (^) primitive
❌ tedious, complex
LM ❌ prone to error
program
questions (^) solutions
**Train Test** (^) existing
methods
❌ need extra training
❌ complex pipeline
❌ irreusable tools
LM
function
large
toolbox
questions
program
solutions
LM
questions
program
solutions
Figure 1.OurTROVEinduces reusable functions to produce better
program solutions than thePRIMITIVEsetting, without training,
supervision, or iterations required byEXISTING METHODs.
programs can also be difficult to verify, as the users may
need to check every operation as well as their combina-
tions and interactions across the entire program (e.g., the
primitive solution in Figure 2).
When human developers are faced with an analogous situ-
ation, theycreate application-specific functions, i.e. tools,
composing primitive functions that are often used together.
For instance, in Figure 2 (right), thecalcrateofchange
tool is easier to understand and less error-prone to use, hence
enabling a more concise and accurate solution.
A few recent works have attempted to use LMs toautomati-
cally induce toolsin a similar way (EXISTING METHODs in
Figure 1). However, existing methods tend to either induce
large and ponderous toolboxes and/or have added complex-
ity and data requirements. For instance, Qian et al. (2023)
propose CREATOR to disentangle planning (tool making)
from execution, but at the cost of producing hundreds or
more tools that are challenging for models to reuse or hu-
mans to verify. Further, compared to the standard setting
(PRIMITIVEin Figure 1) that only needs test data to produce
their solutions, Wang et al. (2023a) propose life-long learn-
ing via an automatic curriculum, equipped with iterative

# arXiv:2401.12869v1 [cs.AI] 23 Jan 2024


```
df = pd.DataFrame({
“Year”: [ 2013 , 2014 , 2015 , 2016 , 2017 ],
“Vacation days”: [ 23 , 18 , 11 , 15 , 8 ]
})
```
```
# Calculate the rate of change in values
calc_rate_of_change(df: pd.DataFrame,
value_column: str, time_column: str,
time1: any, time2: any) –> float
```
```
The table shows how many days of vacation import^ pandas^ as^ pd^
Austin had taken each year. What was the
rate of change between 2015 and 2016?
```
```
question primitive functions advanced functions^
```
```
# get the row for each time stamp
row_2015 = df[df[“Year”] == 2015
row_2016 = df[df[“Year”] == 2016
# get the value for each time
value_2015 = row_2015[“Vacation days”].values[0]
value_2016 = row_2015[“Vacation days”].values[0]
# calculate the rate of change
rate = (value_2016 - value_2015) / 2
```
```
primitive solution
```
```
calc_rate_of_change(df, “Vacation
days”, “Year”, 2015 , 2016 )
```
```
tabular environment advanced solution^
```
Figure 2.Function design affect solutions. Using primitive functions results in complex, error-prone solutions (middle), while using
abstract functions leads to more concise and accurate solutions (right).

self-verification. Cai et al. (2023); Yuan et al. (2023) require
additional training and validation datasets to create tools
ahead to be used by solutions, plus auxiliary modules such
as self-verification or toolset deduplication.

In this paper, we proposeTROVE, atraining-free method
of inducing averifiable andefficient function toolbox (§3),
and using these functions to write solutions.TROVEfea-
tures three major components: using and growing a toolbox
maintained over time, execution agreement-based selection,
and periodic toolbox trimming. Notably, our methodre-
quires zero additional training or supervision, and selects
programs only by their inter-execution agreement. Given a
stream of questions,TROVEproduces their solutions, along
with inducing a handy toolbox to solve the questions.

We experiment on 11 datasets from three real-world tasks
(§4): (1) mathematical problems with the MATH dataset,
(2) table question answering on TabMWP, WTQ, and HiTab,
and (3) compositional visual reasoning with GQA. Com-
pared to baselines usingCODELLAMAas well as previous
state-of-the-art methods usingGPT, ourTROVEconsis-
tently produces solutions with higher accuracy and reduced
complexity, while maintaining a significantly smaller, effi-
cient function library (§5). We further show that via human
study (§6), verifying solutions generated byTROVEis31%
faster and13%more accurate, than solutions generated by
baseline methods.

### 2. Problem Statement & Baseline Methods

We formally define the task of problem solving via programs
(§2.1) and introduce corresponding baseline methods (§2.2).

2.1. Problem Solving via Programs

We focus on problems that are describable in natural lan-
guage (NL) and solvable using programs. Concretely, given
an examplex, i.e., an NL queryqgrounded on an en-
vironmente, we ask a language modelLM to write a
programmatic solutionsby composing multiple functions

```
F={f 1 ,⋯,fn}. This process is denoted asPLM(q,e,f).
fand hencescan be executed oneto obtain the final an-
swer. We use Python as the programming language for our
experiments, since it is general-purpose thus allowing flexi-
ble functions to be created for most tasks. Each solutionsis
a Python program, and each functionfis a Python function.
It is crucial to note that the difficulty of solution generation
is greatly affected by the usability of the functions in the set
F. Relatively speaking, we can categorize functions into
two types: (i)primitive functions, which only support basic
operations such as Python standard libraries, and (ii)com-
posed functions, which perform more complex operations
by composing multiple basic operations.
```
```
Primitive Functions Primitive functions are atomic, low-
level operations on the task environment, such as subtraction
```
- and division /in theprimitive solutionin Figure 2 (mid-
dle). They are often easy to obtain without expert knowledge
about the application domain. Yet often, to solve a question
as in Figure 2 (left), it requires complex compositions of nu-
merous functions and hence becomes extremely error-prone.
In this example, due only to a tiny mistake, which calls
the wrongrow 2015 instead ofrow 2016 , the output goes
wrong despite all the other steps being correct.

```
Composed Functions Composed functions, such as the
calcrateofchangein Figure 2 (right), combine various
primitive functions. Conceptually, they areeasier to use, as
they align better with the actions asked in the question. In
this example, it is easier to associate “What is the rate of
change ...” with a function namedcalcrateofchange,
compared to certain compositions of data slicingdf[⋅],
check equality==, get valuecell.values[index], sub-
traction-, and division/. Practically, composed functions
areless error-prone, by only showing an API interface and
abstracting intricate details inside.
Composed functions are often crafted by human experts, by
recognizing and generalizing shared functionality. However,
this process is costly and hardly scalable to new domains.
```

```
Therefore, it is crucial to create such functions automatically,
to enhance problem solving while saving human labor.
```
```
2.2. Baseline Methods
```
```
We introduce two baselines that generate programs using
primitive and composed functions. All methods, including
our main method later in§3, operate solely by prompting
an LM, and do not update the parameters of the LM itself.
```
```
Using Primitive Functions Our first baseline,PRIMI-
TIVE, asks models to generate programs using primitive
functions, which is the de facto approach for program-aided
problem solving without tool induction (Cheng et al., 2023;
Gao et al., 2023b).
As exemplified in Figure 3, our prompt inputs consist of
four components: (1) an NL instruction specifying the task,
(2) the function signature and textual docstring of primitive
functions,^1 (3)c(example, solution) pairs to demonstrate
the usage of primitive functions, and lastly (4) the query
and environment (q,e) of the current testing example. We
collect the code snippets from model responses assand
execute oneto obtain the final result and evaluate it. Please
find more detailed examples in §A.
```
```
# import the pandas library
import pandas as pd
```
```
Instruction Toolbox
You task is to ... ...
```
```
Question What is the median of vacation days?
Environment
```
```
Solution
df = pd.DataFrame({"Year": [ 2014 , 2015 ],
"Vacation days": [ 18 , 11 ]})
avg_days = df["Vacation days"].mean()
```
```
Year Vacation days
2014 18
2015 11
```
```
Question What is the rate of change from
Monday to Tuesday?
```
```
1 2
```
```
3
in-context pairs
```
```
4
```
```
test
```
```
example
```
**Environment** (^) Day _Price ($)
Monday_ 200
_Friday_ 300
| Year | Vacation days |
| -------- | ---------------------- |
| 2014 | 18 |
| 2015 | 11 |
| Day | _Price ($)_ |
| ------------- | ------------ |
| Monday | 200 |
| Tuesday | 300 |
inputexample
outputsolution
Figure 3.Example input prompt on the tabular environment. We
textualize tables by markdown format in the prompt, and ask LMs
to parse tables intopandasDataFrame in program solutions.
Abstracting Functions Example Wise Our second base-
lineINSTANCEasks models to create tools for each example,
and use them in the solution for that example. Qian et al.
(2023) found this two-step process helpful for model rea-
soning by disentangling tool abstraction (planning) from
example-wise decision (execution).
(^1) We do not show the built-in functions since LMs have already
been trained on them extensively. We only have 1-2 primitives in
addition (§4.3, §4.4), so they can easily fit into the prompt.
In preliminary studies, we compared generating functions
and solutions in two sequential responses or prompting the
model to generate both in one response. They performed
comparably, so we adopted the latter approach since it is
simpler. Concretely, given a set of primitive functionsP,
for each example (q,e), the model needs to generate the
functionsFby composing operations inP, as well as the
solutionsthat usesF. Grounding on Figure 4,Fis the
induced functionsnippet, andsis thegenerated solution.

##### &

calc_rate_of_change(df, “Vacation days”, “Year”,
2015 , 2016 )

```
Solution generation
```
```
def calc_rate_of_change(
df, value_column, time_column, time1, time2):
‘‘‘Calculate the rate of change in values’’’
row1 = df[df[time_column] == time
row2 = df[df[time_column] == time
value1 = row1[value_column].values[ 0 ]
value2 = row2[value_column].values[ 0 ]
return (value2 - value1) / (time2 - time1)
```
```
Function induction
```
```
question environment
```
```
LM
```
##### &

```
calc_rate_of_change(df, “Vacation days”, “Year”,
2015 , 2016 )
```
```
generated solution
```
```
def calc_rate_of_change(
df, value_column, time_column, time1, time2):
‘‘‘Calculate the rate of change in values’’’
row1 = df[df[time_column] == time
row2 = df[df[time_column] == time
value1 = row1[value_column].values[ 0 ]
value2 = row2[value_column].values[ 0 ]
return (value2 - value1) / (time2 - time1)
```
```
induced function
```
```
question environment
```
```
LM
```
```
input
```
```
output
```
```
Figure 4.An illustration of a model inducing a composed function
and using it to generate the solution at the same time.
```
```
We include the same four prompt components used inPRIM-
ITIVE, but alter the instruction and example outputs accord-
ing to theF&sgeneration format. We query each example
independently to allow example-wise function induction.
However, it is not possible to share these functions across
examples, even if they have similar functions.
```
### 3. Inducing Reusable Functions On-the-fly

```
Now, we present our main methodTROVEthat: uses and
grows a toolbox iteratively over time (§3.1), selects optimal
outputs based on execution agreement (§3.2), and periodi-
cally trims low-utility functions from the toolbox (§3.3).
```
```
3.1. Using and Growing the Toolbox
```
```
To learn functions that can be reused across examples, we
maintain a shared function libraryFover time. To keep
our method running in linear time, we process all exam-
plesonlinein a streaming fashion. We start withF=∅
and gradually add or remove functions from it. Figure 5
illustrates the processing of examplextat timet.
First, we define 3 modes with which LMs can interact with
the current toolboxF
t
```
. 1 IMPORT: In this mode, the
LM is instructed to import defined functions from the tool-
box and apply them to solvex
t
. 2 CREATE: In this
mode, the LM is instructed to create a new functionfnewt
and add it to the toolbox. 3 SKIP: In this mode, the LM


is instructed to only use primitive functions just as in the
PRIMITIVEsetting.

For each example, for each of the three modes we sam-
ple from the LM to generateK responses, for a total
of 3 Kresponses per example.^2 Each response contains
a solutionsand functionf as shown in Figure 4, ex-
cept for theSKIPmode that only containss. We de-
note these candidate responses as(fmti,stim), wherem∈
{IMPORT,CREATE,SKIP}andi∈{ 1 ,⋯,K}.

```
question environment
```
```
IMPORT CREATE SKIP
```
```
&
```
## x K

```
&
```
## x K

```
&
```
## x K

```
solution function(s)
```
```
discard
insuccess
```
```
generate
```
```
execution
```
```
&
```
```
&
&
```
```
&
&
```
```
&^
group w/ highest
output agreement
```
```
group by
outputs
```
```
7
13
4 rank by #ops^
the best
candidate
```
```
in 3 modes
```
```
function
library
add to
```
```
use
trim
```
Figure 5.TROVEillustration. Top: generate solutions while using
and growing the toolbox. Bottom: select the best response by
execution agreement. Left: periodically trim low-utility functions.

3.2. Agreement-Based Selection

From the 3 Ksampled(f,s)pairs, we select one to use via
self-consistency (Shi et al., 2022; Li et al., 2022a; Wang
et al., 2023b) and solution complexity. We first execute all
solutions and remove those that cannot execute successfully.
Next, we select an answer based on the consistency of the
execution outputs, keeping the solutions that produce the
most frequently occurring answer. Next, if there are multiple
solutions with the same frequency, we rank solutions by the
number of operations they require, and pick the one with
the least operations (preferring simple solutions). Finally, if
multiple candidates remain, we break the tie by arbitrarily
choosing the one that appears first in the model prediction
list. We add the function in this best response to the toolbox,
and adopt its solution as the solution for the current example.

3.3. Periodic Toolbox Trimming

Not all the functions induced by models are highly reusable.
Hence, we also propose to periodically trim the toolbox to

(^2) In preliminary studies, we tried to let models choose one mode
and only generate in that mode, but results degraded significantly.
effectively remove low-utility functions.
Periodically during testing, we remove functions that have
been used less thanλtimes. By observing that function-
usage frequency has a logarithmic relation with data size,
we setλ=C×log 10 (n), whereC=^12 ,nis the number of
examples processed so far.^3

### 4. Testbeds: Program-Solvable Tasks

```
We now introduce the three programmatic tasks for experi-
ments: math problems, table question answering, and visual
reasoning. We first state the default primitive functions
(§4.1), then describe the specialized functions for each task.
```
```
4.1. The Default Primitive Functions
```
```
For all experiments, we instruct models to generate solutions
as Python programs, so that default primitive functions are
built-in Python functions.
4
We use these default primitives
for MATH (§4.2), and add other data-related functions for
TableQA (§4.3) and VisualQA (§4.4).
```
```
4.2. Math Problems
To test model abilities in solving math problems, we use the
MATH (Hendrycks et al., 2021) dataset that covers ques-
tions from seven subjects: algebra, counting and probability,
geometry, intermediate algebra, number theory, prealgebra,
and precalculus. Table 1 lists the number of test examples,
since our methods only need test data. We only use the
default primitives, i.e., built-in Python functions.
```
```
Task Dataset Size Primitive Functions
```
```
MATH
```
```
algebra 881
```
```
built-in functions
```
```
count & prob. 291
geometry 237
inter. algebra 503
number theory 497
prealgebra 636
precalculus 156
```
```
TABLEQA
```
```
TabMWP 5,376 +pandas
WTQ 4,344 +pandas
```
```
HiTab 1,574 ++pandasparsetable
```
```
VISUALQA GQA 12,
```
```
+PIL.Image
+locateobjects
+visualqa
+cropregion
```
```
Table 1.Statistics and primitives for three tasks.
```
(^3) To ensure all examples can be solved by functions available
in the library, we update the solutions for examples previously
using trimmed functions (<5%of the dataset), by re-generating
solutions under IMPORT & SKIP modes.
(^4) https://docs.python.org/3/library/functions.html


```
Method Metric MATH TABLEQA VISUAL
alg count geo inte num prealg precal TabMWP WTQ HiTab GQA
```
```
PRIMITIVE
```
```
acc ↑ 0.15 0.14 0.06 0.05 0.16 0.21 0.10 0.43 0.20 0.09 0.
# ops↓ 15.4 10.9 15.1 17.0 12.3 12.1 20.8 17.4 24.3 16.5 24.
# lib ↓ — — —
```
```
INSTANCE
```
```
acc ↑ 0.22 0.23 0.07 0.06 0.23 0.26 0.17 0.36 0.17 0.12 0.
# ops↓ 18.4 10.2 26.8 28.2 14.3 10.6 26.9 8.3 8.4 14.1 18.
# lib ↓ 39 7 36 82 5 16 36 3,175 537 31 395
```
```
TROVE
```
```
acc ↑ 0.25 0.26 0.08 0.11 0.25 0.29 0.17 0.47 0.21 0.18 0.
# ops↓ 18.8 10.0 25.4 23.9 11.2 11.7 19.6 10.9 9.2 9.3 20.
# lib ↓ 10 1 7 8 8 4 7 10 11 5 7
```
```
Table 2.CODELLAMA-7B-INSTRUCTresults on MATH, TABLEQA, and VISUALtasks.
```
```
4.3. Table Question Answering
```
```
We adopt three table question answering datasets: TabMWP
(Lu et al., 2023), WTQ (Pasupat & Liang, 2015), and Hitab
(Cheng et al., 2022). They cover a diverse range of question
types and table structures. We represent tables aspandas
DataFrame objects since this is a standard table library to
use with Python; we accordingly add thepandaslibrary
into the set of primitive functions for the table QA task.
```
```
TabMWP The TabMWP dataset (Lu et al., 2023) includes
math word problems on relational tables. Questions in
TabMWP are relatively simple, such as performing numeri-
cal calculations (e.g., “What is the mean of the numbers?”)
and argument selection (“Who has the most OBJECT?”).
We directly input the serialized tables in markdown format
in model prompts, because they are relatively small.
```
WTQ The WikiTableQuestions (WTQ) dataset (Pasupat
& Liang, 2015) contains questions about semi-structured
Wikipedia tables, which feature un-normalized cell values,
thus requires string processing (e.g., parse the number from
“$ 100.00”) and external knowledge retrieval (e.g., find the
country name of “Franco Pellizotti (ITA)”) operations.
Because WTQ tables can be too long to input limits, we put
DataFrame previews in the prompt, and instruct models to
usepandas.readtableto load tables from CSV files.

```
HiTab The Hitab dataset (Cheng et al., 2022) contains
questions about hierarchical matrix tables. HiTab tables
have more complex structures and require special operations
such as multi-hop selection along a bi-dimensional header
hierarchy. We similarly load HiTab tables from the source
JSON files using theparsetablefunction used by Cao
et al. (2023), and add it as the primitive functions.
```
```
4.4. Visual Reasoning
```
```
We use the GQA dataset (Hudson & Manning, 2019) that
contains real-world images and compositional questions
```
```
about them. However, the skills required to solve GQA
questions are more advanced than the previous two tasks.
Although image processing libraries such asPILorcv2are
available, our preliminary experiments show that using these
libraries alone is extremely hard for models to generate vi-
able solutions (only achieving 1% accuracy), not to mention
further inducing advanced functions.
Therefore, we adopt three main primitive actions from Vis-
Prog (Gupta & Kembhavi, 2022) of two types: (1) neural
modules:visualqa,locateobjects; and (2) processing
modules:cropregion; along with the image loading mod-
ulePIL.Image. To reduce the extent of expert engineering,
we did not adopt the other six actions used in VisProg, since
they may be tailored to GQA queries or crafted shortcuts
to assist models (e.g.,checkexists). Removing these
actions slightly degrades model performance.
```
### 5. Experiments

```
We introduce the experiment setup (§5.1) and evaluation
metrics (§5.2), then report and analyze the results (§5.3).
```
```
5.1. Experimental Setup
We compare our methodTROVE(§3) to the two base-
linesPRIMITIVEandINSTANCE(§2.2). We mainly use
CODELLAMA2-7B-INSTRUCTfor experiments,^5 but also
useGPT-4to fairly compare with existing SOTA methods.
We includec= 2 examples in prompts, sampledK= 5
responses in each mode, and trim the toolbox every 200
steps. By default, we set the decoding temperature to 0. 6
and use top-p 0. 95. We limit the model to generate at most
512 tokens to prevent excessive hallucination and save com-
putational cost. To accommodate for randomness in the
sampling result, we run each experiment five times and re-
port the best-performing run. For all methods, we evaluate
the results on test examples.
```
(^5) We are the first to show that open-source LMs can make tools.


5.2. Evaluation Metrics

We propose three metrics to comprehensively evaluate gen-
erated solutions and induced functions.

Answer Correctness (acc↑) The most practically impor-
tant aspect of solutions is correctness. We measure if the
execution outcome of the solution program exactly matches
the ground-truth answer(s).

Solution Complexity (# ops↓) We also measure the pro-
gram complexity by counting the number of function calls
involved. Solutions with fewer functions are easier and
quicker to understand and verify.

Library Size (# lib↓) It is important to control the library
size and encourage function sharing across examples. Com-
pared to multiple tools performing similar operations, fewer
tools with distinct functionaly enable easier tool selection
during solution generation. Since the number of primitive
functions is always the same on a given dataset, we only
report the number of additionally induced functions.

5.3. Model Performance

Table 2 shows the results on all datasets.TROVEproduces
the most accurate solutions with generally lower complexity,
while maintaining a small, efficient function library.

Math Problems Compared toPRIMITIVE, TROVEsub-
stantially improves answer correctness by 30–120% across
7 datasets. Comparing toINSTANCE, TROVEyields 8.7–
83.3% higher correctness using 60.0–90.2% fewer tools.

Table Question Answering Notably,INSTANCEyields
lower correctness thanPRIMITIVEon most datasets, while
generating hundreds even thousands of functions. We con-
jecture the reason to be increased task difficulty and dataset
size (thanMATH), driving up the number of functions
and confusing the model with too many low-utility options.
WhileTROVE, by reusing and trimming functions, allevi-
ates this distraction and gives the best results.

Visual Question Answering TROVEstill scores the best,
butINSTANCEperforms substantially worse than other meth-
ods. with a correctness drop of 0. 21 compared toPRIMI-
TIVE. Similarly toTABLEQA, a larger number of functions
(i.e., 395) are created, which greatly challenges the solu-
tion generation process. Based on our result analysis, we
conjecture that it is difficult to create many valid reusable
functions for GQA, hence most induced functions are in-
valid and impair solution generation.

```
5.4. Comparing with Other Tool-Making Methods
```
```
In addition, we compareTROVEwith three existing meth-
ods that perform tool making, namely LATM (Cai et al.,
2023), CREATOR (Qian et al., 2023), and CRAFT (Yuan
et al., 2023). Notably, all three methods include extra mod-
ules or supervision not required by our method, and were
only demonstrated effective onclosed-sourceGPT models.
Specifically, LATM requires an extra training set to induce
tools in advance, and a validation set to verify the tools.
CREATOR runs multiple iterations of decision, execution,
and rectification. CRAFT requires extra training data and
tool retrieval modules, and necessitates GPT models.
To make a fair comparison, we also useGPT-4with our
TROVEapproach, and test on three datasets — algebra
fromMATH, TabMWP fromTABLEQA, and GQA from
VISUALQAtask — that overlap with these works. Due to
resource limitations, we do not experiment on all 11 datasets.
We believe the result differences in these 2 datasets are
representative to demonstrate the superiority of our method.
```
```
Method accMATH↑ algebra# lib↓ accTabMWP↑ # lib↓ acc↑GQA# lib↓
w/ additional supervision
LATM 0.30 - 0.09 - 0.29 -
CRAFT 0.68 282 0.88 181 0.45 525
w/ additional rectification & iteration
Creator 0.65 875 0.81 4,595 0.34 -
w/o supervision, rectification, or iteration
TROVE 0.72 16 0.92 38 0.44 8
```
```
Table 3.Comparing with existing methods usingGPT-4. We adopt
the baseline results as reported in Yuan et al. (2023). We do not
report thecomplexitymetric since none of these methods report it
(our results in Table 2).
```
```
In Table 3,TROVEoutperforms these existing state-of-
the-art methods on all datasets and most evaluation as-
pects, while having a much simpler pipeline.TROVEnot
only works with closed-sourceGPTs, but also open-source
CODELLAMA(Table 2), with which CRAFT reported near-
random performance using their method.
```
```
Training Advantage of Seen Primitive Functions While
GPT-4outperformsCODELLAMAon most tasks, it is im-
pressive to see that they perform comparably on the GQA
task, as shown in Table 4. We conjecture that this difference
between GQA and other tasks comes from the advantage
of models using corresponding primitive functions. For ex-
ample,pandas, as a primitive inTABLEQA, may appear
frequently in the training data, so models may be more
proficient in or inclined to write solutions with this library.
In contrast, models may have never seen any data using
GQA primitives since no large-scale data are annotated
```

```
with these hand-crafted functions. The difficulty of models
learning these primitives in context is similar to that of
learning the induced functions. While GQA reduces GPT-
4’s advantage in using primitives, it is intriguing to observe
that 7BCODELLAMA2 performs on par withGPT-4by
making and (re-)using tools.
```
```
Model Method accEvaluation Metrics↑ # ops↓ # lib↓
```
```
CODELLAMA PTRIMITIVEROVE 0.370.44 24.620.3 - 7
```
```
GPT-4 PTRIMITIVEROVE 0.400.44 27.420.2 - 8
```
```
Table 4.7BCODELLAMA2 andGPT-4perform comparably on
the GQA task without training advantage.
```
### 6. Efficient Verification by Humans

```
Model-produced solutions may not be reliable, so we investi-
gate ifTROVEfacilitates more efficient solution verification
by humans. To test this hypothesis, we randomly selected
100 examples and asked 6 human evaluators to verify the cor-
rectness of solutions generated byPRIMITIVE, INSTANCE,
and TROVE on the WTQ dataset.^6
We evaluate human performance from two aspects. (1)
Detection accuracy: if they can accurately predict whether
or not the solution is correct; the higher the better, and
(2) Time used: how many seconds they need to verify an
average example; the lower the better.
Table 5 shows the results. Foraccuracy, using tools in solu-
tions (INSTANCEandTROVE) improves detection accuracy
by 13.0–14.3%, compared to usingPRIMITIVEfunctions
only. For thetime used, our method reduces the average
time by 31 .4%compared toPRIMITIVE, and 43 .0%com-
pared toINSTANCE. However, using irreusable tools (i.e.,
theINSTANCEsetting) actually increases the time by 20 .4%.
Overall,TROVEsubstantially speeds up the verification
process, while achieving similar detection accuracy. See§C
for the test results of individual participants.
```
```
Method Accuracyavg std↑ avgTime (s)std↓
```
```
PRIMITIVE 0.77 0.109 25.5 6.
INSTANCE 0.88 0.024 30.7 12.
TROVE 0.87 0.057 17.5 4.
```
```
Table 5.Human accuracy and time in verifying model-produced
solutions with three methods experimented.
```
(^6) We chose WTQ from theTABLEQAtask, becauseTABLEQA
functions are more complex than those in other tasks, and WTQ is
the fairest representative of the task with mid-level difficulty.

### 7. Inducing Specialized Functions

```
TROVEdemonstrates its generality on multiple tasks and
datasets. In this section, we further show thatTROVEcan
produce specialized functions that (1) differ in forms across
tasks, and (2) differ in functions across datasets. We use the
CODELLAMA2-7B results as an example.
```
```
Different Function Forms Across Tasks We com-
pare the three tasks and list a few exemplar func-
tions in Table 6. For MATH, the model often im-
ports external libraries (sympy) to enable using ad-
vanced functions, or creates functions targeting certain
problems (calculateremainder). TABLEQAtasks in-
duce more complex functions comprising many primi-
tive functions (e.g.,getmatchaftercondition). VI-
SUALQAfunctions involve fewer primitives, for exam-
ple, getimageregion is a chain of two primitives:
locateobjectsandcropregion.
```
```
Task Example Functions
```
```
MATH def product^ calculate_remainder= 1 (numbers, modulus):^
for number in numbers: product *= number
return produce % modulus
```
```
from sympy import solve
```
```
def find_range(df: pd.DataFrame, column: str) -> int:
"""Find the range of values in the column."""
# get the min and max values
min_value = df[value_column].min()
max_value = df[value_column].max()
# calculate the range
range_value = max_value - min_value
return range_value
def get_match_after_condition(
df, condition_column: str, condition: any,
value_column: str) -> any:
""""Get the match that comes after the match that
satisfies a condition in the specified column."""
row = df[df[condition_column] == condition]
index = row.index[ 0 ] + 1
if index < len(df):
return df.iloc[index][value_column]
else:
return None
```
```
from PIL import Image
from toolbox import crop_region, locate_objects
def get_object_region(
image: Image.Image, object_name: str
) -> Image.Image:
"""Locate the crop the image of the object."""
boxes = locate_objects(image, object_name)
object_image = crop_region(image, boxes)
return object_image
```
```
TABLEQA
```
```
def calculate_remainder(numbers, modulus):
product = 1
for number in numbers: product *= number
return produce % modulus
```
```
from sympy import solve
```
```
def find_range(df: pd.DataFrame, column: str) -> int:
"""Find the range of values in the column."""
# get the min and max values
min_value = df[value_column].min()
max_value = df[value_column].max()
# calculate the range
range_value = max_value - min_value
return range_value
def get_match_after_condition(
df, condition_column: str, condition: any,
value_column: str) -> any:
""""Get the match that comes after the match that
satisfies a condition in the specified column."""
row = df[df[condition_column] == condition]
index = row.index[ 0 ] + 1
if index < len(df):
return df.iloc[index][value_column]
else:
return None
```
```
from PIL import Image
from toolbox import crop_region, locate_objects
def get_object_region(
image: Image.Image, object_name: str
) -> Image.Image:
"""Locate the crop the image of the object."""
boxes = locate_objects(image, object_name)
object_image = crop_region(image, boxes)
return object_image
```
```
VISUALQA
```
def calculate_remainder(numbers, modulus):
product = 1
for number in numbers: product *= number
return produce % modulus

from sympy import solve

```
def find_range(df: pd.DataFrame, column: str) -> int:
"""Find the range of values in the column."""
# get the min and max values
min_value = df[value_column].min()
max_value = df[value_column].max()
# calculate the range
range_value = max_value - min_value
return range_value
def get_match_after_condition(
df, condition_column: str, condition: any,
value_column: str) -> any:
""""Get the match that comes after the match that
satisfies a condition in the specified column."""
row = df[df[condition_column] == condition]
index = row.index[ 0 ] + 1
if index < len(df):
return df.iloc[index][value_column]
else:
return None
```
```
from PIL import Image
from toolbox import crop_region, locate_objects
def get_object_region(
image: Image.Image, object_name: str
) -> Image.Image:
"""Locate the crop the image of the object."""
boxes = locate_objects(image, object_name)
object_image = crop_region(image, boxes)
return object_image
```
```
Table 6.Example functions induced by TROVE on three tasks.
```
```
Varied Functionalities Across Datasets ForMATHand
TABLEQAthat have multiple datasets, we further analyze
the variance in functions between the datasets.
AmongMATHdatasets, as shown in Figure 6, core func-
tions such asmathandsympyoverlap. Meanwhile, some
functions are particularly useful for certain questions, such
as the self-definedcalculuateremainderfornumber the-
oryquestions, andsympy.Polygenforgeometryquestions.
Figure 7 shows some functions in threeTABLEQAdatasets.
```

```
number
```
```
precalculus
```
```
prealgebra
```
```
geometry
```
```
intermediate
algebra
```
```
algebra
```
```
math
itertools
```
```
sympy.solve
```
```
math.gcd
```
```
sympy.Eq
sympy.abc.x
```
```
math.floor
```
```
sympy
```
```
numpy
```
**_counting_** (^) calculate_remainder
count_digits
sympy.gcd
operatorstatistics (^)
math.cos
cmath
sympy.Polygon math.radians
math.pi
sympy.integrate (^)
math.ceil (^) sympy.lambdify
sympy.symbols
sympy.Matrix
sympy.solvers.solve
Figure 6.Illustration of MATH libraries for seven subjects.
Due to the greater variance in question types and table struc-
tures, most functions differ except for the basicpandas.
**_TabMWP_**
pandas
calculate_rate_of_change
find_range
find_median
find_mode
find_difference
sum_values
calc_total_cost
... ...

#### WTQ

```
count_by_condition
get_value_by_condition
```
... ... (^) get_next_match

#### HiTab

```
parse_table
```
```
get_data_cell
```
```
get_most_common
```
```
... ...
```
Figure 7.Illustration of three TABLEQA function libraries.

Overall,TROVEcan effectively propose both functions that
are (1) generic to the task, and (2) specific to each domain.
These induced functions not only help solve the problems,
but also characterize their functional distribution.

### 8. Ablation Studies

We conduct ablation studies to test the robustness to ordering
(§8.1) and the importance of toolbox trimming (§8.2).

8.1. Robustness to Ordering

Our method inputs examples in a streaming fashion and
orders examples as they appear in the original dataset. How-
ever, it is important to study if variations in example order-
ing would affect final results. We select one dataset from
each task (MATHalgebra, HiTab, GQA) as representatives
for this examination.

We shuffle the examples five times with different random
seeds and runTROVEon each of the five orderings. We
report the range of metric values and their standard deviation
in Table 7. As a reference, we denote results (from§3) using
the original dataset order asoriginal.

For all three datasets, no significant variance exists between
theoriginalandrandomizedordering — theoriginalresults
well within therandomizedvalue range, and standard devia-
tions are small — showing thatTROVEis robust to example

```
Method / Value acc↑Evaluation Metrics# ops↓ # lib↓
```
```
MATHalgebra
original 0.25 18.8 10
value range 0.23–0.24 17.3–19.0 5–
std.dev. 0.000 0.879 1.
HiTab
original 0.18 9.3 5
value range 0.17–0.18 9.0–9.9 8–
std.dev. 0.003 0.358 0.
GQA
original 0.43 20.6 6
value range 0.43–0.44 20.4–20.6 6–
std.dev. 0.005 0.150 0.
```
```
Table 7.CODELLAMAresults with alternative orders.
```
```
ordering. While the datasets may not be ordered in a way
that optimizes function induction, theoriginalordering may
just be another instance of somewhatrandomizedordering.
```
```
8.2. Without Toolbox Trimming
Periodic function trimming is crucial to ensure the efficiency
ofTROVE. To demonstrate this point, we compare to a
TROVEversion without toolbox trimming. In Figure 8,
when including the trimming mechanism, the size of func-
tion libraries significantly decreases by 74% - 90%. The
accuracy and complexity in Table 8 also slightly degraded.
```
```
Method MATHacc↑ algebra# ops↓ acc↑HiTab# ops↓ acc↑GQA# ops↓
without trim 0.25 19.6 0.15 10.5 0.39 21.
with trim 0.25 18.8 0.18 9.3 0.44 20.
```
```
Table 8.TROVE results with and without toolbox trimming.
```
```
While the trimming threshold and time interval are easily
adjustable, one can flexibly keep more functions to explore
more diverse functions, or fewer due to certain constraints.
```
### 9. Related Work

```
Generating Program Solutions Many works focus on
generating Python programs to solve problems, such as
math (Ni et al., 2023; Li et al., 2022b; Gao et al., 2023b;
Chen et al., 2022) and table QA (Cao et al., 2023; Cheng
et al., 2023). Yet most programs are built with basic oper-
ations or libraries (e.g.,sum,pandas), and may be tedious
and erroneous. Gupta & Kembhavi (2022); Subramanian
et al. (2023); Sur ́ıs et al. (2023) generate image-executable
programs by hand-crafting task-specific functions, Gao et al.
(2023a); Yang et al. (2023) extend this to audio and video
```

```
Figure 8.Library size without toolbox trimming.
```
modalities, but still require expert designs from humans. In
contrast, our work enjoys the benefit of advanced functions
with reduced human labor by inducing functions using LMs.

Domain-Specific Library Abstraction Shin et al. (2019)
mine common code idioms and utilize them for program
synthesis. Ellis et al. (2023) propose to induce functions
bottom-up from a large corpus via a wake-sleep Bayesian
process. Wong et al. (2021) improve the search efficiency,
and Bowers et al. (2023) proposed a top-down method
STITCH to save memory. Most recently, LILO (Grand et al.,
2023) integrates LLMs into STITCH and abstract libraries
with auto-documentation. While these methods all work on
domain-specific logical forms, running them with general-
purpose languages may vastly enlarge the search space, thus
have limited applicability on many real-world tasks. Instead,
our method generates general-purpose Python programs and
can readily extend to new tasks.

Making Program Tools Using LLMs With the advances
of LLMs, many works explore using LLMs to build tools.
Cai et al. (2023) examine homogenous BigBench tasks,
where in each task all examples use a single tool. Qian
et al. (2023) work on math and table QA tasks but create
numerous tools that are not re-used across tasks – this serves
as ourINSTANCEbaseline. Yuan et al. (2023) increase
tool sharing via additional training but still yield redundant
tools. Xin et al. (2023) enables a growing lemma library
for math theorem proving, but requires external supervision
from the theorem prover and expert heuristics. Wang et al.
(2023a) can build and learn skills in the embodied Minecraft
world, yet requires self-verification and iterative refinement.
In comparison, our method leverages execution agreement
without any training or supervision.

### 10. Conclusion

We proposedTROVE, a method for inducing a toolbox of
reusable functions to use in solving programmatic tasks.
TROVEproduces simpler and more accurate solutions than

```
programming language DreamCoder PATOIS STITCH LILO TROVE
domain-specific ✓✓✓✓
general purpose ✓
tool making modules LATM Creator CRAFT Voyager TROVE
training / curriculum ✓✓✓
self-verification ✓✓✓✓
iterative refine ✓✓✓
self-consistency ✓
```
```
Table 9.Modules required by existing methods and our TROVE.
```
```
existing methods, using sufficiently smaller function li-
braries. Moreover, it facilitates human program verification
to be31%faster and13%more accurate. Finally,TROVE
can induce diverse functions across tasks and datasets, shed-
ding insights on data-specific characteristics.
```
### Acknowledgements

```
We thank Shuyan Zhou and Zhoujun Cheng for the insight-
ful discussions about this work, Saujas Vaduguru for pro-
viding feedback about the draft, and all human participants
of the verification study.
```
### References

```
Artzi, Y. and Zettlemoyer, L. Weakly Supervised Learn-
ing of Semantic Parsers for Mapping Instructions to
Actions. Transactions of the Association for Compu-
tational Linguistics, 2013. URL https://doi.org/10.1162/
tacla00209.
```
```
Bowers, M., Olausson, T. X., Wong, L., Grand, G., Tenen-
baum, J. B., Ellis, K., and Solar-Lezama, A. Top-down
synthesis for library learning. Proc. ACM Program.
Lang., 7(POPL), jan 2023. doi: 10.1145/3571234. URL
https://doi.org/10.1145/3571234.
```
```
Cai, T., Wang, X., Ma, T., Chen, X., and Zhou, D.
Large language models as tool makers. arXiv preprint
arXiv:2305.17126, 2023. URL https://arxiv.org/pdf/2305.
17126.
```
```
Cao, Y., Chen, S., Liu, R., Wang, Z., and Fried, D. Api-
```

```
assisted code generation for question answering on varied
table structures.arXiv preprint arXiv:2310.14687, 2023.
URL https://arxiv.org/pdf/2310.14687.
```
Chen, W., Ma, X., Wang, X., and Cohen, W. W. Program
of thoughts prompting: Disentangling computation from
reasoning for numerical reasoning tasks.arXiv preprint
arXiv:2211.12588, 2022.

Cheng, Z., Dong, H., Wang, Z., Jia, R., Guo, J., Gao, Y.,
Han, S., Lou, J.-G., and Zhang, D. HiTab: A hierarchical
table dataset for question answering and natural language
generation. InProceedings of the 60th Annual Meeting
of the Association for Computational Linguistics (Volume
1: Long Papers), May 2022.

Cheng, Z., Xie, T., Shi, P., Li, C., Nadkarni, R., Hu, Y.,
Xiong, C., Radev, D., Ostendorf, M., Zettlemoyer, L.,
Smith, N. A., and Yu, T. Binding language models
in symbolic languages. InThe Eleventh International
Conference on Learning Representations, 2023. URL
https://openreview.net/forum?id=lH1PV42cbF.

Ellis, K., Wong, L., Nye, M., Sable-Meyer, M., Cary, L.,
Anaya Pozo, L., Hewitt, L., Solar-Lezama, A., and Tenen-
baum, J. B. Dreamcoder: growing generalizable, inter-
pretable knowledge with wake–sleep bayesian program
learning.Philosophical Transactions of the Royal Society
A, 381(2251):20220050, 2023.

Gao, D., Ji, L., Zhou, L., Lin, K. Q., Chen, J., Fan, Z., and
Shou, M. Z. Assistgpt: A general multi-modal assistant
that can plan, execute, inspect, and learn.arXiv preprint
arXiv:2306.08640, 2023a.

Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang,
Y., Callan, J., and Neubig, G. Pal: Program-aided lan-
guage models. InInternational Conference on Machine
Learning, pp. 10764–10799. PMLR, 2023b.

Grand, G., Wong, L., Bowers, M., Olausson, T. X., Liu,
M., Tenenbaum, J. B., and Andreas, J. Lilo: Learning
interpretable libraries by compressing and documenting
code.arXiv preprint arXiv:2310.19791, 2023.

Gupta, T. and Kembhavi, A. Visual programming: Compo-
sitional visual reasoning without training.arXiv preprint
arXiv:2211.11559, 2022. URL https://arxiv.org/pdf/2211.
11559.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart,
S., Tang, E., Song, D., and Steinhardt, J. Measuring
mathematical problem solving with the math dataset.
arXiv preprint arXiv:2103.03874, 2021. URL https:
//arxiv.org/pdf/2103.03874.

```
Hudson, D. A. and Manning, C. D. Gqa: A new dataset for
real-world visual reasoning and compositional question
answering. InProceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pp. 6700–
6709, 2019.
```
```
Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J.,
Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago,
A., et al. Competition-level code generation with alpha-
code.Science, 378(6624):1092–1097, 2022a.
```
```
Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J.,
Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago,
A., et al. Competition-level code generation with alpha-
code.Science, 378(6624):1092–1097, 2022b.
```
```
Liang, P., Tripp, O., and Naik, M. Learning minimal abstrac-
tions. InProceedings of the 38th annual ACM SIGPLAN-
SIGACT symposium on Principles of programming lan-
guages, pp. 31–42, 2011.
```
```
Lu, P., Qiu, L., Chang, K.-W., Wu, Y. N., Zhu, S.-C., Ra-
jpurohit, T., Clark, P., and Kalyan, A. Dynamic prompt
learning via policy gradient for semi-structured math-
ematical reasoning. arXiv preprint arXiv:2209.14610,
```
2023. URL https://arxiv.org/pdf/2209.14610.

```
Majumder, B. P., Mishra, B. D., Jansen, P., Tafjord, O.,
Tandon, N., Zhang, L., Callison-Burch, C., and Clark,
P. Clin: A continually learning language agent for
rapid task adaptation and generalization.arXiv preprint
arXiv:2310.10134, 2023.
```
```
Ni, A., Iyer, S., Radev, D., Stoyanov, V., Yih, W.-t., Wang,
S., and Lin, X. V. Lever: Learning to verify language-
to-code generation with execution. InInternational Con-
ference on Machine Learning, pp. 26106–26128. PMLR,
2023.
```
```
Pasupat, P. and Liang, P. Compositional semantic parsing
on semi-structured tables. InProceedings of the 53rd
Annual Meeting of the Association for Computational
Linguistics and the 7th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers).
Association for Computational Linguistics, July 2015.
URL https://aclanthology.org/P15-1142.
```
```
Qian, C., Han, C., Fung, Y. R., Qin, Y., Liu, Z., and Ji, H.
Creator: Disentangling abstract and concrete reasonings
of large language models through tool creation. arXiv
preprint arXiv:2305.14318, 2023. URL https://arxiv.org/
pdf/2305.14318.
```
```
Shi, F., Fried, D., Ghazvininejad, M., Zettlemoyer, L., and
Wang, S. I. Natural language to code translation with
execution. InProceedings of EMNLP. Association for
Computational Linguistics, December 2022. URL https:
//aclanthology.org/2022.emnlp-main.231.
```

Shin, R., Allamanis, M., Brockschmidt, M., and Polozov,
O. Program synthesis and semantic parsing with learned
code idioms, 2019.

Subramanian, S., Narasimhan, M., Khangaonkar, K., Yang,
K., Nagrani, A., Schmid, C., Zeng, A., Darrell, T., and
Klein, D. Modular visual question answering via code
generation. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Volume 2:
Short Papers), July 2023.

Sur ́ıs, D., Menon, S., and Vondrick, C. Vipergpt: Visual
inference via python execution for reasoning. arXiv
preprint arXiv:2303.08128, 2023.

Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu,
Y., Fan, L., and Anandkumar, A. Voyager: An open-
ended embodied agent with large language models.arXiv
preprint arXiv:2305.16291, 2023a.

Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi,
E. H., Narang, S., Chowdhery, A., and Zhou, D. Self-
consistency improves chain of thought reasoning in lan-
guage models. InThe Eleventh International Confer-
ence on Learning Representations, 2023b. URL https:
//openreview.net/forum?id=1PL1NIMMrw.

Wong, C., Ellis, K. M., Tenenbaum, J., and Andreas, J.
Leveraging language to learn program abstractions and
search heuristics. In Meila, M. and Zhang, T. (eds.),
Proceedings of the 38th International Conference on Ma-
chine Learning, volume 139 ofProceedings of Machine
Learning Research, pp. 11193–11204. PMLR, 18–24 Jul

2021. URL https://proceedings.mlr.press/v139/wong21a.
html.

Xin, H., Wang, H., Zheng, C., Li, L., Liu, Z., Cao, Q.,
Huang, Y., Xiong, J., Shi, H., Xie, E., et al. Lego-prover:
Neural theorem proving with growing libraries. arXiv
preprint arXiv:2310.00656, 2023.

Yang, Z., Li, L., Wang, J., Lin, K., Azarnasab, E., Ahmed,
F., Liu, Z., Liu, C., Zeng, M., and Wang, L. Mm-react:
Prompting chatgpt for multimodal reasoning and action.
arXiv preprint arXiv:2303.11381, 2023.

Yin, P. and Neubig, G. A syntactic neural model
for general-purpose code generation. arXiv preprint
arXiv:1704.01696, 2017.

Yuan, L., Chen, Y., Wang, X., Fung, Y. R., Peng, H.,
and Ji, H. Craft: Customizing llms by creating and
retrieving from specialized toolsets. arXiv preprint
arXiv:2309.17428, 2023.

Zettlemoyer, L. and Collins, M. Online learning of relaxed
ccg grammars for parsing to logical form. InProceedings
of the 2007 Joint Conference on Empirical Methods in

```
Natural Language Processing and Computational Natu-
ral Language Learning (EMNLP-CoNLL), pp. 678–687,
2007.
```

### A. Prompt Example

We introduced baseline methods (PRIMITIVEandINSTANCE) in§2.2 and our main method in§3. To more concretely
illustrate the prompts beyond textual description, we provide figure examples.

Figure 9 is an example prompt used in thePRIMITIVEsetting, where on the bottom is the current test example andSolution
is expected to be filled by the model.

```
# from PIL import Image
from PIL import Image
# Load object bounding boxes in the image
locate_objects(image: Image, object_name: str) -> list
# Answering basic visual question with neural models
visual_qa(image: Image, question: str) -> str
# Crop the specified box region from the image
crop_region(image: Image, boxes: list) -> Image
```
```
Toolbox
```
```
Instruction
Your task is to use tools, i.e., Python functions, to reason over the image.
Write a program solution to the question by decomposing it into multiple steps. Then specify the
tools used in each step by importing from the toolbox. Make sure that all tools used in the solution
are defined in the toolbox. Do not use any undefined functions.
All images are presented as PIL.Image objects, you can use functions in PIL, cv2 if they help.
```
```
Question Who is carrying the umbrella?
Image “data/gqa/testdev_images/n10052.jpg”
Solution
image_file = "data/gqa/testdev_images/n100552.jpg"
image = Image.open(image_file).convert('RGB')
carry_boxes = locate_objects(image, "carry umbrella")
carry_region = crop_region(image, carry_boxes)
answer = visual_qa(image=carry_region, question="Who is carrying the umbrella?")
print(answer)
Tools
from toolbox import Image, locate_objects, crop_region, visual_qa
```
**Question** (^) Are there either any small refrigerators or microwaves in the picture?
**Image** “data/gqa/testdev_images/n579256.jpg”
**Solution**
Figure 9.Example prompt in the primitive setting.
We use the same format in the INSTANCEsetting, but changed theInstructionto be:
Your task is to write the solution with high-level tools, i.e., Python functions, to reason over the image.
Think about the potential program solution for this example, you can create high-level functions, that could be used to solve this
example. For example, if the solution involves multiple actions that are always used together, it is more efficient to create and use
the tool.
Similarly, in the ONLINEsetting, respectively for theCREATE,IMPORT, andSKIPmodes, theInstructions reads:
Your task is to write Python program solutions to reason over images. You should also create Python functions that can be used by
your solution, if you believe the function can be reused to solve other questions.
Your task is to write Python program solutions to reason over images. The toolbox section lists all the available functions that can
be used in your solution.
Your task is to write Python program solutions to reason over images.


### B. Evaluation Metric Details

Answer Correctness We measure if the solution execution result matches the annotated answers. For answers that are
expected in a textual format (e.g., “brown”), we use the Exact Match (EM) metric; for numerical answers, we convert it to
float type and round to two decimals, then measure if the values match (with a difference less than1e− 6 ).

Program Complexity We quantify the complexity of programs in their number of operations. Concretely, we separate
solutions into multiple expressions, and parse each expression into abstract syntax trees (AST). We take the depth of each
AST as the number of operations conducted in the corresponding expression. We sum this value of all expressions and
denote it as the complexity (i.e., number of operations) of the entire program.

### C. Human Study: Individual Results

In the verification human study, performance between people may vary due to their programming expertise. Therefore, in
this section, we present more detailed results of individual participants, and justify the significance of our findings.

```
Figure 10.Individual verification accuracy and time used.
```
As shown in Figure 10 (left), verification accuracy is higher for tool-involved methods (TROVEandINSTANCE) compared
to using primitive functions only (PRIMITIVE). Most people perform verification more accurately on programs produced by
TROVE except one person (the red line), which also has lower detection accuracy in general.

Their times used for verification are shown on the right, with the same line color identifying each human. Most people
findTROVEthe fastest,PRIMITIVEtaking 8. 2 more seconds on average, andINSTANCErequires another 10. 8 seconds.
However, one person responded differently and personally foundINSTANCEmore efficient thanPRIMITIVE, which is the
major contribution of the relatively large variance of INSTANCEreported in §6.


