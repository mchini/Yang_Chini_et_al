# Anesthetics uniquely decorrelate hippocampal network activity, alter spine dynamics and affect memory consolidation

This repository contains code related to the paper "Anesthetics fragment hippocampal network activity, alter spine dynamics and affect memory consolidation".
The manuscript is available here: https://www.biorxiv.org/content/10.1101/2020.06.05.135905v1


<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="121px" viewBox="-0.5 -0.5 121 61" content="&lt;mxfile host=&quot;app.diagrams.net&quot; modified=&quot;2021-01-22T07:16:51.025Z&quot; agent=&quot;5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36&quot; etag=&quot;67EtAC5bQ1YEDOa7LWV7&quot; version=&quot;14.2.6&quot; type=&quot;google&quot;&gt;&lt;diagram id=&quot;Kg9Y_M0CIAsYhnU6arRw&quot; name=&quot;Page-1&quot;&gt;jZJNT4QwEIZ/TY8mQBX0qLirJnowbLLGi+nSkdYUSkoR8NdbZLpANpuYcJh55qPDO0NoWvYPhtXiRXNQJAp4T+g9iaLwMorI+AV8mEiSJBMojOSYNINM/gDCAGkrOTSrRKu1srJew1xXFeR2xZgxulunfWq1frVmBZyALGfqlO4lt2Ki11fBzB9BFsK/HAYYKZlPRtAIxnW3QHRDaGq0tpNV9imoUTyvy1S3PRM9Dmagsv8pMPH+LYnfb2/cnp4+Xnfbw/PXBXb5ZqrFH95BY3FgO3gVjG4rDmOjgNC7TkgLWc3yMdq5vTsmbKmcFzoTW4Kx0J+dNTwq4E4HdAnWDC4FC2iMouHV0AT9bt5B6IUVC/19HcO1F8fWszLOQHG8Oy/hL7Y4Zbr5BQ==&lt;/diagram&gt;&lt;/mxfile&gt;" onclick="(function(svg){var src=window.event.target||window.event.srcElement;while (src!=null&amp;&amp;src.nodeName.toLowerCase()!='a'){src=src.parentNode;}if(src==null){if(svg.wnd!=null&amp;&amp;!svg.wnd.closed){svg.wnd.focus();}else{var r=function(evt){if(evt.data=='ready'&amp;&amp;evt.source==svg.wnd){svg.wnd.postMessage(decodeURIComponent(svg.getAttribute('content')),'*');window.removeEventListener('message',r);}};window.addEventListener('message',r);svg.wnd=window.open('https://viewer.diagrams.net/?client=1&amp;page=0');}}})(this);" style="cursor:pointer;max-width:100%;max-height:61px;"><defs/><g><rect x="0" y="0" width="120" height="60" fill="#ffffff" stroke="#000000" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject style="overflow: visible; text-align: left;" pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 118px; height: 1px; padding-top: 30px; margin-left: 1px;"><div style="box-sizing: border-box; font-size: 0; text-align: center; "><div style="display: inline-block; font-size: 12px; font-family: Helvetica; color: #000000; line-height: 1.2; pointer-events: all; white-space: normal; word-wrap: normal; ">Test</div></div></div></foreignObject><text x="60" y="34" fill="#000000" font-family="Helvetica" font-size="12px" text-anchor="middle">Test</text></switch></g></g><switch><g requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"/><a transform="translate(0,-5)" xlink:href="https://www.diagrams.net/doc/faq/svg-export-text-problems" target="_blank"><text text-anchor="middle" font-size="10px" x="50%" y="100%">Viewer does not support full SVG 1.1</text></a></switch></svg>

Code to align calcium imaging recordings from different sessions but with the same FOV is in the [Alignment](https://github.com/mchini/Yang_Chini_et_al/tree/master/Alignment%20Scripts%20(Python)) folder.

Before alignment             |  After alignment
:-------------------------:|:-------------------------:
![](no_alignment.gif)  |  ![](with_alignment.gif)

Python code for analysis of calcium transients and correlation matrices is in [Figures 3-5](https://github.com/mchini/Yang_Chini_et_al/tree/master/Figures%203-5%20(Python)) folder

![](correlations_small.png)

Matlab code for clustering in the spatial and temporal domain and sleep classification is in the [Figures 5-7](https://github.com/mchini/Yang_Chini_et_al/tree/master/Figures%205-7%20(MATLAB)) folder

![](clustering.png)

Further Matlab code that was used for the ephys-part of the paper can be found in this other [repository](https://github.com/mchini/HanganuOpatzToolbox)

![](ephys_small.png)

R scripts and datasets that were used for all statistical analysis are available in the [Stats](https://github.com/mchini/Yang_Chini_et_al/tree/master/Stats%20(R)) folder.

Raw 2-photon and electrophysiology data is available at this [repository](https://gin.g-node.org/SW_lab/Anesthesia_CA1/) on GIN.
