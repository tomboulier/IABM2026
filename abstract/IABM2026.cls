
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IABM2026}[2025/11/05 Classe sobre académique pour la conférence IABM 2026]

\LoadClass[a4paper,12pt]{article}

% Encodage et langue
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
% Note: requires texlive-lang-french package
% sudo apt-get install texlive-lang-french
\IfFileExists{french.ldf}{%
  \RequirePackage[french]{babel}%
}{%
  \RequirePackage{babel}%
  \PackageWarning{IABM2026}{french.ldf not found, using default language}%
}

% Packages standards
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{caption}
\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}
\RequirePackage{xcolor}
\RequirePackage{url}
\RequirePackage{hyperref}

% Mise en page
\geometry{margin=2.5cm}
\setstretch{1.1}
\setlength{\parskip}{0.8em}
\setlength{\parindent}{0pt}

% Style des titres
\titleformat{\section}
  {\normalfont\Large\bfseries}
  {\thesection}{1em}{}

\titleformat{\subsection}
  {\normalfont\bfseries}
  {\thesubsection}{1em}{}

% Définition des métadonnées
\newcommand{\IABMtitle}[1]{\def\@IABMtitle{#1}}
\newcommand{\IABMauthors}[1]{\def\@IABMauthors{#1}}
\newcommand{\IABMaffiliations}[1]{\def\@IABMaffiliations{#1}}
\newcommand{\IABMstatus}[1]{\def\@IABMstatus{#1}}

% Page de titre IABM 2026
\renewcommand{\maketitle}{
  \begin{center}
    {\LARGE \textbf{\@IABMtitle}}\\[1em]
    {\large \@IABMauthors}\\[0.5em]
    \vspace{0.5em}
    {\small \@IABMaffiliations}\\[0.5em]
    \textit{\@IABMstatus}    
  \end{center}
}

% En-têtes et pieds de page sobres
\pagestyle{fancy}
\fancyhf{}
\lhead{Conférence IABM 2026}
\rhead{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Style des légendes
\captionsetup{
  font=small,
  labelfont=bf,
  labelsep=period
}

% Bibliographie par défaut
\RequirePackage{natbib}        % gestion des citations
\setcitestyle{numbers,comma,compress}  % citations numériques compressées [1,2,3]
\AtBeginDocument{%
  \providecommand{\bibsection}{\section*{Références}} % titre Références
}

\endinput
