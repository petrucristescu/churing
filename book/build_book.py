# -*- coding: utf-8 -*-
"""
Builds "One Hundred Years of Invisible Spaces" as a PDF using fpdf2.
Body font: Arial. Math/symbol fallback: Segoe UI Symbol.
"""
import os
from fpdf import FPDF

FONTS = r"C:\Windows\Fonts"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "One_Hundred_Years_of_Invisible_Spaces.pdf")

# Color palette
INK      = (28, 28, 34)
ACCENT   = (60, 70, 120)
LIGHT    = (120, 120, 130)
RULE     = (200, 200, 210)
BOX_BG   = (242, 243, 248)
BOX_BAR  = (60, 70, 120)


class Book(FPDF):
    def __init__(self):
        super().__init__(format="A4", unit="mm")
        self.set_auto_page_break(True, margin=22)
        self.set_margins(28, 24, 28)
        # Body family
        self.add_font("Body", "",  os.path.join(FONTS, "arial.ttf"))
        self.add_font("Body", "B", os.path.join(FONTS, "arialbd.ttf"))
        self.add_font("Body", "I", os.path.join(FONTS, "ariali.ttf"))
        self.add_font("Body", "BI", os.path.join(FONTS, "arialbi.ttf"))
        # Symbol fallback (math glyphs missing from Arial)
        self.add_font("Sym", "", os.path.join(FONTS, "seguisym.ttf"))
        self.set_fallback_fonts(["Sym"])
        self._show_footer = True

    def footer(self):
        if not self._show_footer:
            return
        self.set_y(-15)
        self.set_font("Body", "I", 8)
        self.set_text_color(*LIGHT)
        # No number on front matter pages (page < 3)
        if self.page_no() > 2:
            self.cell(0, 8, str(self.page_no() - 2), align="C")

    # ---- building blocks -------------------------------------------------
    def h_part(self, kicker, title):
        self.add_page()
        self.ln(40)
        self.set_text_color(*ACCENT)
        self.set_font("Body", "B", 13)
        self.cell(0, 8, kicker.upper(), align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_text_color(*INK)
        self.set_font("Body", "B", 26)
        self.multi_cell(0, 12, title, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(6)
        # decorative rule
        self.set_draw_color(*RULE)
        w = 40
        x = (self.w - w) / 2
        self.line(x, self.get_y(), x + w, self.get_y())

    def h_chapter(self, num, title, people=None):
        if self.get_y() > self.h - 80:
            self.add_page()
        self.ln(6)
        self.set_text_color(*ACCENT)
        self.set_font("Body", "B", 11)
        self.cell(0, 7, num.upper(), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*INK)
        self.set_font("Body", "B", 19)
        self.multi_cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        if people:
            self.set_text_color(*LIGHT)
            self.set_font("Body", "I", 11)
            self.multi_cell(0, 6, people, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_draw_color(*RULE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)
        self.set_text_color(*INK)

    def para(self, text):
        self.set_x(self.l_margin)
        self.set_font("Body", "", 11.5)
        self.set_text_color(*INK)
        self.multi_cell(0, 6.4, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2.6)

    def core_idea(self, text):
        """Highlighted 'core idea' callout box."""
        self.ln(2)
        pad = 4
        self.set_font("Body", "BI", 11.5)
        x0 = self.l_margin
        w = self.w - self.l_margin - self.r_margin
        # measure height
        y_start = self.get_y()
        # render into a dummy to compute lines: use split_only
        lines = self.multi_cell(w - 2 * pad - 3, 6.2, text, split_only=True)
        h = len(lines) * 6.2 + 2 * pad
        if y_start + h > self.h - self.b_margin:
            self.add_page()
            y_start = self.get_y()
        # background
        self.set_fill_color(*BOX_BG)
        self.rect(x0, y_start, w, h, style="F")
        # left bar
        self.set_fill_color(*BOX_BAR)
        self.rect(x0, y_start, 1.6, h, style="F")
        # text
        self.set_xy(x0 + pad + 3, y_start + pad)
        self.set_text_color(*ACCENT)
        self.multi_cell(w - 2 * pad - 3, 6.2, text)
        self.set_y(y_start + h)
        self.ln(4)
        self.set_text_color(*INK)


# =====================================================================
# CONTENT
# =====================================================================

P = "para"
C = "core"

prologue = [
    (P, "Open any physics textbook from three different centuries and a strange thing happens. The eighteenth-century mathematician studying how heat spreads through an iron bar reaches for the same machinery as the nineteenth-century engineer describing a vibrating string, and both of them, without knowing it, are writing down the equations that a twentieth-century physicist will use to describe an electron that has no position at all. The notation drifts. The vocabulary changes. But underneath, the same mathematical object keeps surfacing, like a shape glimpsed again and again through different windows."),
    (P, "This book is about that recurring shape. It is the story of how, over roughly one hundred years, mathematicians and physicists slowly came to suspect that the world is not really described in the three-dimensional space we move through, but in vast, abstract spaces — spaces whose “points” are entire functions, whose “distances” are integrals, and whose “directions” are waves. These are the invisible spaces of the title."),
    (P, "We tend to imagine that physics is written in the space of everyday experience: here is a particle, there is a force, this is how far it travels. The deeper truth, the one that took a century to articulate, is stranger:"),
    (C, "Physics is not written in physical space, but in abstract spaces of functions."),
    (P, "The question that organizes everything that follows is simple to ask and hard to answer: why do these invisible spaces keep reappearing? Why should the mathematics of heat have anything to do with the mathematics of a hydrogen atom? The answer, when it finally arrived in the hands of David Hilbert and John von Neumann, was that all of these problems had secretly been the same problem all along — different views of a single geometric structure that no one had been able to see directly. This is the path we will trace, from Newton's falling apple to the rotating phases of a quantum computer."),
]

parts = [
    {
        "kicker": "Part I",
        "title": "The Birth of Continuous Thought",
        "chapters": [
            {
                "num": "Chapter 1",
                "title": "Newton and Leibniz: Motion Becomes Calculus",
                "people": "Isaac Newton · Gottfried Wilhelm Leibniz",
                "body": [
                    (P, "Before calculus, change was something you described after the fact. You could say a cannonball had travelled so far in so many seconds, but you could not say how fast it was moving at one precise instant, because an instant has no duration and motion seemed to require time to pass. Newton and Leibniz, working independently in the 1660s and 1670s, broke this impasse by inventing a mathematics of the infinitely small."),
                    (P, "Newton thought in terms of fluxions — quantities flowing in time, with the derivative as the rate of that flow. Leibniz thought in terms of differentials, infinitesimal increments dx and dy, and gave us the elegant notation dy/dx and the integral sign ∫ that we still use today. The two men quarrelled bitterly over priority, but they had reached the same summit by different faces of the same mountain."),
                    (P, "The two operations they discovered are mirror images. The derivative answers “how fast is this changing right now?” The integral answers “how much has accumulated in total?” The Fundamental Theorem of Calculus, the crown jewel of the era, says these are inverse operations: accumulation undoes rate-of-change. For the first time, humanity had a reliable language for continuous infinity — for processes that are smooth, unbroken, and infinitely divisible."),
                    (P, "This was more than a computational trick. It was the first crack in the idea that mathematics describes static shapes. From now on, mathematics could describe becoming. And once you can talk about functions and their rates of change, you are only a few steps away from asking what a function really is — and that question will eventually carry us out of ordinary space entirely."),
                ],
            },
            {
                "num": "Chapter 2",
                "title": "Cauchy: The Need for Rigor",
                "people": "Augustin-Louis Cauchy",
                "body": [
                    (P, "Calculus worked spectacularly well, and for over a century almost no one worried that its foundations were a logical scandal. Infinitesimals were quantities that were somehow both zero and not-zero; infinite sums were manipulated as if they always added up to something sensible. The philosopher George Berkeley mocked these infinitesimals as “the ghosts of departed quantities.” He had a point."),
                    (P, "In the early nineteenth century, Cauchy set out to exorcise the ghosts. His key move was to replace the vague idea of an infinitely small quantity with the precise idea of a limit: a value that a process approaches as closely as we like, even if it never exactly arrives. A derivative was no longer a ratio of ghosts but the limit of ordinary ratios. An infinite sum was no longer a mystical total but the limit of its partial sums."),
                    (P, "This forced a new and crucial question into the open: when does an infinite process actually converge to something? Some infinite sums settle down to a definite value; others wander off to infinity or oscillate forever. Cauchy gave careful criteria for telling the difference, and in doing so he made convergence — not the individual terms, but the destination of the whole sequence — the central object of analysis."),
                    (C, "A sequence can have a destination even when no single step ever reaches it."),
                    (P, "This idea of completeness — that a well-behaved space should contain the limits of its own convergent sequences — looks like a technicality. It is in fact the seed of Hilbert space. The whole machinery of quantum mechanics will eventually rest on the demand that certain infinite sequences of functions are guaranteed to converge to a function that is still inside the space. Cauchy did not know this. He was simply trying to make calculus honest."),
                ],
            },
        ],
    },
    {
        "kicker": "Part II",
        "title": "Waves Enter Mathematics",
        "chapters": [
            {
                "num": "Chapter 3",
                "title": "Fourier: Heat Becomes Frequency",
                "people": "Joseph Fourier",
                "body": [
                    (P, "In 1807 Joseph Fourier submitted a paper on how heat diffuses through solids. The committee judging it included Lagrange, one of the greatest mathematicians alive, and they were scandalized. Fourier had claimed something that sounded absurd: that any function describing a temperature distribution — even a jagged, discontinuous one — could be written as a sum of smooth sine and cosine waves."),
                    (P, "The heat equation itself was tractable for a single wave: a sinusoidal temperature profile simply decays over time, the sharper ripples fading fastest. Fourier's bold leap was to handle any initial temperature by decomposing it into waves, evolving each wave separately, and adding the results back together. To make this work he needed his impossible claim — that the space of “reasonable” functions is spanned by sines and cosines."),
                    (P, "He was essentially right, and the implications run far deeper than heat. Fourier had discovered that a function carries a hidden second identity. A musical chord can be described moment by moment as a wiggling pressure wave, or equivalently as a recipe of pure frequencies — so much of this note, so much of that one. These are two descriptions of the same object, related by the Fourier transform that converts between them."),
                    (C, "Every function has a twin: its frequency spectrum. Position and frequency are two languages for one thing."),
                    (P, "Notice the quiet revolution. Fourier is treating an entire function as a single thing to be analyzed — something you can decompose, like a vector resolved into components along axes. The “axes” here are the waves. This is the first time functions start to behave like points in a space with directions. The dual relationship between a profile and its spectrum will return, transfigured, as the relationship between a quantum particle's position and its momentum."),
                ],
            },
            {
                "num": "Chapter 4",
                "title": "Euler: The Circle Hidden in Growth",
                "people": "Leonhard Euler",
                "body": [
                    (P, "We step back half a century to Euler, because his discovery is the engine that makes everything later run. Euler studied the exponential function eˣ, the unique function that is its own rate of change — the mathematics of unchecked growth, of compound interest and populations. He also studied the imaginary unit i, the square root of −1, an object invented to solve equations that had no real solutions."),
                    (P, "Euler asked what happens when you feed an imaginary number into the exponential. The answer, Euler's formula, is one of the most astonishing sentences in mathematics: e^{ix} = cos x + i·sin x. Growth, when pointed in the imaginary direction, does not grow at all. It rotates."),
                    (C, "Rotation is exponential growth aimed in the imaginary direction. The circle and the exponential are the same motion seen from two angles."),
                    (P, "Set x = π and you get the famous identity e^{iπ} + 1 = 0, binding together five fundamental constants. But the real prize is the geometric picture. The expression e^{ix} traces the unit circle as x increases; it is a point spinning steadily around the origin at constant speed. The single number x simultaneously encodes an angle, a phase, and a frequency."),
                    (P, "Hold onto this image of steady rotation. When Schrödinger writes down how a quantum state changes in time, the entire evolution will turn out to be exactly this — a phase e^{-iEt/ħ} rotating in an abstract space. Euler, studying pure mathematics with no thought of physics, had already written down the gear that drives the quantum world."),
                ],
            },
        ],
    },
    {
        "kicker": "Part III",
        "title": "Functions Become Geometry",
        "chapters": [
            {
                "num": "Chapter 5",
                "title": "Riemann and Cantor: Infinite Structures",
                "people": "Bernhard Riemann · Georg Cantor",
                "body": [
                    (P, "By the mid-nineteenth century, mathematics was ready to take infinity seriously as an object rather than a process. Two men pushed hardest. Bernhard Riemann reimagined geometry itself, showing that space need not be flat or three-dimensional; one could define curved spaces of any number of dimensions purely by specifying how to measure distance within them. Geometry was freed from physical intuition — a space was now anything with a consistent notion of distance and angle."),
                    (P, "Georg Cantor went further and asked how big infinity is. His shocking discovery was that there are different sizes of infinity: the infinity of whole numbers is strictly smaller than the infinity of points on a line. Infinity was not a single vague vastness but a structured hierarchy that could be reasoned about with precision. Cantor's set theory gave mathematicians the courage to treat collections of infinitely many objects — including infinite collections of functions — as legitimate things."),
                    (P, "Between them, Riemann and Cantor delivered two permissions that the next generation would seize. Riemann said: dimension is not sacred; you may have as many as you need, even infinitely many. Cantor said: the infinite is not forbidden; you may build whole worlds out of infinite sets. Once functions could be gathered into infinite sets and a notion of distance could be imposed on them, the stage was set for a genuinely new kind of geometry — a geometry whose points are functions."),
                ],
            },
            {
                "num": "Chapter 6",
                "title": "Hilbert: The Discovery of Function Space",
                "people": "David Hilbert",
                "body": [
                    (P, "Around 1900, David Hilbert and his school pulled all of these threads together. The decisive idea was deceptively modest: treat a function as a vector. We are used to vectors as little arrows with components — (3, 4) in the plane, say. A function f(x) can be thought of the same way, except it has infinitely many components, one value f(x) for each input x. It is an arrow in a space of infinitely many dimensions."),
                    (P, "To make this a geometry rather than a mere analogy, Hilbert needed the geometric notions of length and angle. He found them in the integral. The inner product of two functions, written ⟨f, g⟩, is the integral of their product — a single number measuring how much the two functions overlap, exactly as the dot product measures how much two arrows point the same way. From the inner product flow everything else: the “length” of a function, the angle between functions, and most importantly the notion of two functions being perpendicular, or orthogonal."),
                    (C, "Functions behave like vectors: they have length, they have angles between them, and some are perpendicular to others."),
                    (P, "Now Fourier's old miracle snaps into focus. The sine and cosine waves are mutually orthogonal — perpendicular axes in function space. Writing a function as a Fourier series is nothing but resolving a vector into its components along these axes. What had looked like a strange analytic accident was revealed as ordinary geometry, merely carried out in infinitely many dimensions."),
                    (P, "Hilbert added one final requirement, Cauchy's old demand for completeness: every convergent sequence of functions must have its limit inside the space. A complete inner-product space of this kind is now called a Hilbert space. It is, quite literally, infinite-dimensional geometry. Hilbert built it to study integral equations. Within two decades it would be revealed as the home address of all of physics."),
                ],
            },
        ],
    },
    {
        "kicker": "Part IV",
        "title": "The Quantum Break",
        "chapters": [
            {
                "num": "Chapter 7",
                "title": "Schrödinger: Waves of Matter",
                "people": "Erwin Schrödinger",
                "body": [
                    (P, "In 1926 Erwin Schrödinger proposed that matter, like light, has a wave nature, and that an electron in an atom should be described not by a position but by a wavefunction — written ψ (psi) — spread out over space. Where the classical physicist asked “where is the electron?”, Schrödinger answered with a smooth function whose value at each point encodes how much of the electron's presence is there."),
                    (P, "His equation, the Schrödinger equation, dictates how this wavefunction evolves in time. Structurally it resembles Fourier's heat equation — but with a fateful factor of i, the imaginary unit, sitting in front of the time derivative. That single i changes everything: instead of decaying like heat, the wavefunction's components rotate in phase, exactly along the circle Euler had drawn. The quantum state does not dissipate; it spins."),
                    (P, "Solving the equation for the hydrogen atom, Schrödinger found that only certain wave patterns — standing waves that fit cleanly around the nucleus — are allowed, just as only certain notes resonate on a guitar string. The mysterious quantization of energy, the fact that atoms emit only specific colors of light, fell out automatically as the resonant frequencies of matter waves. Quantization was not an extra rule bolted on; it was what waves do when confined."),
                    (P, "But what was waving? Schrödinger initially hoped ψ was a real, physical ripple. Max Born soon gave the interpretation that stuck: the squared magnitude of the wavefunction, |ψ|², is a probability density — the chance of finding the particle at each location. The wavefunction is complex, living partly in Euler's imaginary direction, and only when squared does it yield the real probabilities we measure. Physics had become irreducibly a theory of functions."),
                ],
            },
            {
                "num": "Chapter 8",
                "title": "Heisenberg: Reality as Algebra",
                "people": "Werner Heisenberg",
                "body": [
                    (P, "A year before Schrödinger, Werner Heisenberg had attacked the same atomic puzzles from a completely different direction — and seemingly arrived at a completely different theory. Heisenberg refused to talk about unobservable things like an electron's orbit. He insisted on building physics only from quantities you can actually measure: the frequencies and intensities of the light atoms emit."),
                    (P, "When he organized these observable quantities into tables of numbers and worked out how to combine them, he stumbled onto a rule that disturbed him: the order of multiplication mattered. Multiplying quantity A by quantity B gave a different answer than B times A. Heisenberg had unknowingly rediscovered matrix multiplication, which mathematicians knew is non-commutative. In his theory the observables — position, momentum, energy — were not numbers at all but matrices, algebraic objects that act on states."),
                    (C, "Observables are not numbers but operators. That A·B ≠ B·A is the mathematical root of the uncertainty principle."),
                    (P, "This non-commutativity is precisely what forbids simultaneously sharp values of position and momentum. The famous uncertainty principle is, at bottom, the statement that the matrices for position and momentum do not commute. Heisenberg's “matrix mechanics” looked nothing like Schrödinger's gentle waves — it was discrete, algebraic, and forbidding. For a brief period physics had two rival theories of the atom, one made of waves and one made of matrices, and no one could see why they should agree. Yet they made identical predictions."),
                ],
            },
        ],
    },
    {
        "kicker": "Part V",
        "title": "The Unification",
        "chapters": [
            {
                "num": "Chapter 9",
                "title": "Dirac and the Language of States",
                "people": "Paul Dirac",
                "body": [
                    (P, "Paul Dirac, reserved and famously precise, supplied the notation and the perspective that let the two rival theories finally be seen as one. He proposed thinking of a quantum system's state as an abstract vector — not a wave in ordinary space, not a column of numbers, but a pure direction in a Hilbert space, independent of how you choose to describe it."),
                    (P, "He invented the bra-ket notation that physicists still breathe today. A state is a “ket”, written |ψ⟩. Its mirror partner is a “bra”, ⟨φ|. Put them together and you get a bracket ⟨φ|ψ⟩ — an inner product, a single complex number measuring the overlap between two states. (The names are Dirac's pun: bra-ket spells bracket.) This is Hilbert's ⟨f, g⟩ wearing a physicist's uniform."),
                    (P, "In this language the structure becomes luminous. States are vectors. Observables are operators that act on them. Measuring an observable asks which special states — eigenstates — are left undisturbed in direction by that operator, and the measured values are the corresponding eigenvalues. Schrödinger's wavefunction ψ(x) is revealed as just the components of the abstract vector |ψ⟩ along the “position” axes; Heisenberg's matrices are just the same operators written out in some chosen basis. The notation did not merely tidy things up — it made the underlying geometry impossible to miss."),
                ],
            },
            {
                "num": "Chapter 10",
                "title": "von Neumann: The Invisible Space Revealed",
                "people": "John von Neumann",
                "body": [
                    (P, "It fell to John von Neumann, in his 1932 masterwork on the mathematical foundations of quantum mechanics, to state plainly what had been emerging: Schrödinger's wave mechanics and Heisenberg's matrix mechanics are not two theories. They are one theory, written in two coordinate systems, describing vectors and operators in the same Hilbert space."),
                    (P, "Schrödinger had been working in the basis of position; Heisenberg in a basis of energy states. A wavefunction and a matrix were the same abstract object expressed along different axes — the way the same arrow has different components depending on how you orient your graph paper. The fierce debate over which picture was “real” dissolved: neither and both. The reality was the basis-independent vector, and the pictures were merely views of it."),
                    (C, "The rival theories were different coordinate systems in the same invisible space."),
                    (P, "Von Neumann did this with full mathematical rigor, supplying the careful theory of operators on Hilbert space that the physicists' bold manipulations had been assuming. In his hands quantum mechanics became a precise piece of geometry: states are unit vectors, time evolution is a rotation that preserves their length, measurement is projection onto an axis, and probabilities are the squared lengths of those projections — Born's rule as pure Pythagoras."),
                    (P, "This is the climax of our hundred-year story. The invisible space that kept flickering into view — in Cauchy's convergent sequences, in Fourier's orthogonal waves, in Hilbert's vector-functions — turned out to be the very stage on which physical reality is enacted. The universe keeps its books not in the space we walk through, but in Hilbert space."),
                ],
            },
        ],
    },
    {
        "kicker": "Part VI",
        "title": "The Fourier Nature of Reality",
        "chapters": [
            {
                "num": "Chapter 11",
                "title": "Position and Momentum as Dual Spaces",
                "people": None,
                "body": [
                    (P, "With the geometric picture in place, Fourier returns transfigured. Recall that every function has a twin, its frequency spectrum, and that the two are related by the Fourier transform. In quantum mechanics this is not a mathematical convenience — it is the relationship between two physical quantities. The wavefunction written in terms of position, ψ(x), and the same state written in terms of momentum, are Fourier transforms of one another."),
                    (P, "Position and momentum are therefore not independent properties that a particle separately possesses. They are two bases for the same Hilbert space, and the change from one to the other is precisely a Fourier transform — a rotation of axes in function space. To know the state fully in one basis is to know it fully in the other; they carry identical information, encoded differently."),
                    (C, "The Fourier transform is a change of basis between the position view and the momentum view of one state."),
                    (P, "And here the uncertainty principle reappears with its true face. There is a hard mathematical fact about Fourier transforms, known long before quantum mechanics: a function and its transform cannot both be sharply concentrated. Squeeze a signal into a brief pulse in time and its frequency content necessarily spreads; pin down a single pure frequency and the signal must extend forever. Translate “time” to position and “frequency” to momentum, and you have Heisenberg's principle exactly. Uncertainty is not a limitation of our instruments. It is a theorem of geometry — the unavoidable price of these two bases being Fourier partners."),
                ],
            },
            {
                "num": "Chapter 12",
                "title": "Euler's Formula as the Engine of Quantum Motion",
                "people": None,
                "body": [
                    (P, "We can now collect the debt we owe to Euler. The Schrödinger equation says that a quantum state of definite energy E evolves in time by multiplication by the phase factor e^{-iEt/ħ}. That is Euler's e^{ix} — a point rotating steadily on the unit circle — with the rotation rate set by the energy. A stationary state does not sit still; it spins in the complex plane, and the higher its energy, the faster it turns."),
                    (P, "So e^{ix} is wearing three hats at once, and quantum mechanics needs all three. It is a wave, the building block of Fourier's decomposition. It is a rotation, a steady turning in the abstract space of states. And it is a frequency, which by Planck's relation E = ħω is energy itself. The same little expression that Euler found by playing with imaginary exponents turns out to be the literal motor of time in the quantum world."),
                    (C, "Schrödinger evolution is phase rotation: every energy state is a hand of Euler's clock, turning at a rate set by its energy."),
                    (P, "An arbitrary state, being a superposition of many energies, is a chord of these rotating phases, each hand of the clock turning at its own speed. The intricate dance of interference — the brightening and darkening of quantum probabilities — is nothing more than these phases falling in and out of step. Strip quantum dynamics to its core and you find Euler's circle, turning."),
                ],
            },
        ],
    },
    {
        "kicker": "Part VII",
        "title": "Modern Consequences",
        "chapters": [
            {
                "num": "Chapter 13",
                "title": "Quantum Computing: Computation in Invisible Spaces",
                "people": None,
                "body": [
                    (P, "If physical reality lives in Hilbert space, then a machine that manipulates Hilbert-space vectors directly would be computing with the grain of the universe rather than against it. That is the idea behind quantum computing, and it is the most concrete payoff of the hundred-year journey. A quantum computer is, quite literally, an engine for steering vectors around in an invisible space."),
                    (P, "A bit in an ordinary computer is 0 or 1. A qubit is a unit vector in a two-dimensional Hilbert space, a superposition that can point anywhere between |0⟩ and |1⟩. With n qubits the state lives in a space of 2ⁿ dimensions — ten qubits already span a thousand-dimensional space, three hundred qubits a space with more dimensions than there are atoms in the observable universe. The exponential vastness that frightened the early set theorists is here a computational resource."),
                    (P, "Quantum logic gates are rotations of this state vector — operations that preserve its length while turning its direction, exactly the length-preserving evolution von Neumann described. A quantum algorithm is a carefully choreographed sequence of such rotations, designed so that Euler's phases interfere: the rotations are arranged to make the amplitudes of wrong answers cancel and the amplitudes of right answers reinforce. Computation becomes interference."),
                    (C, "Qubits are vectors, gates are rotations, and the answer emerges from interference of Euler's phases."),
                    (P, "Reading out the result is von Neumann's measurement — a projection onto an axis, yielding one outcome with a probability equal to the squared length of the projection. Every conceptual ingredient was already present in the 1932 foundations: vectors, rotations, projections, squared amplitudes. The quantum computer is the invisible space made into a machine — the abstract geometry of Hilbert and von Neumann, fabricated in superconducting metal and trapped ions, and put to work."),
                ],
            },
        ],
    },
]

epilogue = [
    (P, "Look back along the path and a single arc emerges. Newton and Leibniz taught us to describe continuous change. Cauchy made that description rigorous and, in demanding convergence, planted the seed of completeness. Fourier showed that a function hides a second self, a spectrum of waves, and quietly began treating functions as objects to be decomposed. Euler, earlier still, had found the rotating phase that would drive everything. Riemann and Cantor licensed dimensions without limit and infinities with structure. Hilbert assembled it all into a geometry whose points are functions."),
    (P, "Then physics walked in and discovered it had been living there the whole time. Schrödinger's waves and Heisenberg's matrices, Dirac's state vectors and von Neumann's rigorous operators — all of them were charts of one territory, the Hilbert space of quantum states. Position and momentum revealed themselves as Fourier-paired bases; uncertainty as a theorem of geometry; time evolution as Euler's phase, turning. And in our own era we have learned to build machines that compute by rotating vectors in that very space."),
    (P, "The chain is almost suspiciously clean: the mathematics of functions became geometry; that geometry became Hilbert space; Hilbert space became the language of quantum mechanics; and quantum mechanics became a new kind of computation. Each link was forged by someone solving a concrete problem — heat, rigor, atoms — with no inkling of where it led."),
    (P, "What, in the end, was actually discovered over these hundred years? Not a new place. We never left the ordinary space of tables and chairs and falling apples. What changed is our understanding of where the laws are written. The deepest description of reality, it turns out, is not a story about objects moving through the space we see, but about vectors turning in a space we cannot."),
    (C, "We never left classical space — we learned that reality is encoded in spaces we cannot see."),
]

# Appendix: symbol dictionary
symbols = [
    ("x, t", "Position and time.",
     "The everyday variables — where something is and when. In a wavefunction ψ(x, t) they are the inputs over which the quantum amplitude is spread.",
     "Arguments of nearly every function in classical and quantum physics."),
    ("n, k", "Counting and wave indices.",
     "n usually labels discrete things — the n-th energy level, the n-th term of a series. k is a wavenumber, counting how many wave cycles fit per unit length (spatial frequency).",
     "Quantum numbers of atomic states (n); plane waves e^{ikx} and Fourier components (k)."),
    ("e", "Euler's number, ≈ 2.71828.",
     "The base of natural growth: the function eˣ is its own derivative. Aimed in the imaginary direction it produces rotation rather than growth.",
     "Exponential decay/growth; and as e^{ix}, the rotating phase at the heart of waves and quantum evolution."),
    ("i", "The imaginary unit, √(−1).",
     "A number whose square is −1. It supplies a second, perpendicular direction to the number line, turning it into the complex plane where rotation lives.",
     "Multiplies the time derivative in the Schrödinger equation; makes the wavefunction complex."),
    ("π", "Pi, ≈ 3.14159.",
     "The ratio of a circle's circumference to its diameter, and so the natural unit of rotation — a full turn is 2π.",
     "Periods of waves; Euler's identity e^{iπ} + 1 = 0; normalization of quantum states."),
    ("∫", "The integral.",
     "A continuous sum — the accumulation of infinitely many infinitesimal pieces. Leibniz's elongated S, for summa.",
     "Total accumulated change; the inner product ⟨f, g⟩; total probability ∫|ψ|² dx = 1."),
    ("∑", "The summation.",
     "A discrete sum over an index. The Greek capital sigma, for sum.",
     "Fourier series; superpositions of discrete quantum states; expansions in a basis."),
    ("ψ (psi)", "The wavefunction.",
     "The complete description of a quantum state as a function over space. Its squared magnitude |ψ|² gives the probability density of finding the particle.",
     "The central object of Schrödinger's wave mechanics."),
    ("φ (phi)", "A phase or a second state.",
     "Often an angle or phase of rotation; also used as a generic second wavefunction or state alongside ψ.",
     "Phase factors e^{iφ}; the bra ⟨φ| in inner products ⟨φ|ψ⟩."),
    ("⟨f, g⟩", "The inner product.",
     "A single number measuring the overlap of two functions or vectors — the infinite-dimensional analogue of the dot product. Computed as an integral of f times g.",
     "Defines length, angle and orthogonality in Hilbert space; the geometric backbone of quantum mechanics."),
    ("|ψ⟩", "A ket (state vector).",
     "Dirac's notation for an abstract quantum state as a vector in Hilbert space, independent of any chosen basis. Its partner ⟨φ| is a bra; together they form a bracket ⟨φ|ψ⟩.",
     "The standard language of quantum states, measurement, and quantum computing."),
    ("ħ (h-bar)", "The reduced Planck constant.",
     "A tiny fundamental constant of nature (h divided by 2π) that sets the scale of quantum effects and converts energy into a rotation rate via E = ħω.",
     "The Schrödinger equation; the phase e^{-iEt/ħ}; the uncertainty principle."),
]


# =====================================================================
# RENDER
# =====================================================================
def build():
    pdf = Book()

    # --- Title page ---
    pdf._show_footer = False
    pdf.add_page()
    pdf.ln(55)
    pdf.set_text_color(*ACCENT)
    pdf.set_font("Body", "B", 13)
    pdf.cell(0, 8, "ONE HUNDRED YEARS OF", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_text_color(*INK)
    pdf.set_font("Body", "B", 40)
    pdf.cell(0, 20, "INVISIBLE SPACES", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_draw_color(*BOX_BAR)
    pdf.line((pdf.w - 60) / 2, pdf.get_y(), (pdf.w + 60) / 2, pdf.get_y())
    pdf.ln(10)
    pdf.set_text_color(*LIGHT)
    pdf.set_font("Body", "I", 15)
    pdf.multi_cell(0, 8, "From Euler's circles to von Neumann's quantum universe", align="C")
    pdf.ln(70)
    pdf.set_font("Body", "", 11)
    pdf.cell(0, 6, "A narrative history of how physics moved into abstract space", align="C")

    # --- Table of contents ---
    pdf.add_page()
    pdf.ln(6)
    pdf.set_text_color(*INK)
    pdf.set_font("Body", "B", 22)
    pdf.cell(0, 12, "Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    def toc_line(label, indent=0, bold=False):
        pdf.set_x(pdf.l_margin + indent)
        pdf.set_font("Body", "B" if bold else "", 11.5)
        pdf.set_text_color(*(ACCENT if bold else INK))
        pdf.cell(0, 7.2, label, new_x="LMARGIN", new_y="NEXT")
    toc_line("Prologue — The Strange Idea", 0)
    for part in parts:
        pdf.ln(1)
        toc_line(part["kicker"] + " — " + part["title"], 0, bold=True)
        for ch in part["chapters"]:
            toc_line(ch["title"], 6)
    pdf.ln(1)
    toc_line("Epilogue — What Was Actually Discovered?", 0, bold=True)
    toc_line("Appendix A — Symbol Dictionary", 0, bold=True)

    pdf._show_footer = True

    # --- Prologue ---
    pdf.h_part("Prologue", "The Strange Idea")
    pdf.add_page()
    pdf.h_chapter("Prologue", "The Strange Idea")
    for kind, txt in prologue:
        (pdf.core_idea if kind == C else pdf.para)(txt)

    # --- Parts & chapters ---
    for part in parts:
        pdf.h_part(part["kicker"], part["title"])
        pdf.add_page()
        for ci, ch in enumerate(part["chapters"]):
            if ci > 0:
                pdf.add_page()
            pdf.h_chapter(ch["num"], ch["title"], ch["people"])
            for kind, txt in ch["body"]:
                (pdf.core_idea if kind == C else pdf.para)(txt)

    # --- Epilogue ---
    pdf.h_part("Epilogue", "What Was Actually Discovered?")
    pdf.add_page()
    pdf.h_chapter("Epilogue", "What Was Actually Discovered?")
    for kind, txt in epilogue:
        (pdf.core_idea if kind == C else pdf.para)(txt)

    # --- Appendix ---
    pdf.h_part("Appendix A", "Symbol Dictionary")
    pdf.add_page()
    pdf.h_chapter("Appendix A", "Symbol Dictionary")
    pdf.para("Every symbol used in this book, with its meaning, the intuition behind it, and where it shows up in physics.")
    for sym, meaning, intuition, where in symbols:
        if pdf.get_y() > pdf.h - 55:
            pdf.add_page()
        pdf.set_font("Body", "B", 15)
        pdf.set_text_color(*ACCENT)
        pdf.cell(0, 9, sym + "    —  " + meaning, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*INK)
        pdf.set_font("Body", "", 11)
        pdf.multi_cell(0, 6, "Intuition.  " + intuition)
        pdf.set_font("Body", "I", 11)
        pdf.set_text_color(*LIGHT)
        pdf.multi_cell(0, 6, "Where it appears.  " + where)
        pdf.ln(4)
        pdf.set_text_color(*INK)

    pdf.output(OUT)
    print("WROTE", OUT, "pages:", pdf.page_no())


if __name__ == "__main__":
    build()
