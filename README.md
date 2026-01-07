# FedCALA
<h2>Reproduing the Results (Table 1)</h2>

<p>
To reproduce the results reported in <strong>Table&nbsp;1</strong> of the project paper,
follow the steps below.
</p>

<h3>Running All Experiments</h3>
<ul>
  <li>Run <code>runner.sh</code>.</li>
  <li>This script executes each specified configuration <strong>5 times</strong> and automatically saves the results.</li>
  <li>After all experiments are completed, run <code>score_aggregator.py</code>.</li>
  <li>This script aggregates the results and produces a table showing the <strong>mean</strong> and <strong>standard deviation</strong> for each configuration.</li>
  <li>The generated table exactly matches <strong>Table&nbsp;1</strong> in the project paper.</li>
</ul>

<h3>Running a Specific Configuration</h3>
<ul>
  <li>If you want to run only a specific configuration, execute <code>train.py</code> directly.</li>
  <li>Before running, modify the desired configuration parameters in <code>openfgl/config.py</code>.</li>
</ul>

<h3>Requirements</h3>
<p>
Install all required dependencies using:
</p>

<pre><code>pip install -r requirements.txt</code></pre>
