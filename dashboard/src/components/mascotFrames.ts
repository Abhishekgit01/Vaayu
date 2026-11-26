// Hand-authored ASCII frames for Monsoon Mascot characters.
// Every string is a fixed-width block meant for a monospace <pre>.

/* ------------------------------------------------------------------ */
/*  MEGH — the cloud with a face                                      */
/* ------------------------------------------------------------------ */

// idle: eyes open
export const MEGH_IDLE: readonly string[] = [
  '         .~~~~~~~~~~.',
  '       .~            ~.',
  '      (    °    °      )',
  '     (       __         )',
  '      (________________)',
] as const

// idle: blink
export const MEGH_BLINK: readonly string[] = [
  '         .~~~~~~~~~~.',
  '       .~            ~.',
  '      (    -    -      )',
  '     (       __         )',
  '      (________________)',
] as const

// sleepy: z Z floating up
export const MEGH_SLEEPY: readonly string[][] = [
  [
    '         .~~~~~~~~~~.    z',
    '       .~            ~.',
    '      (    -    -      )',
    '     (       __         )',
    '      (________________)',
  ],
  [
    '         .~~~~~~~~~~.     Z',
    '       .~            ~.',
    '      (    -    -      )',
    '     (       __         )',
    '      (________________)',
  ],
  [
    '         .~~~~~~~~~~.      z',
    '       .~            ~.',
    '      (    -    -      )',
    '     (       __         )',
    '      (________________)',
  ],
] as const

// drag/mouth open wide
export const MEGH_MOUTH_OPEN: readonly string[] = [
  '         .~~~~~~~~~~.',
  '       .~            ~.',
  '      (    O    O      )',
  '     (       ()         )',
  '      (________________)',
] as const

// thunder: shocked face
export const MEGH_SHOCKED: readonly string[] = [
  '         .~~~~~~~~~~.',
  '       .~            ~.',
  '      (    O    O      )',
  '     (       OO         )',
  '      (________________)',
] as const

/* ------------------------------------------------------------------ */
/*  RAIN FRAMES — cycle these under the cloud                         */
/* ------------------------------------------------------------------ */

export const RAIN_FRAMES: readonly string[][] = [
  [
    "   ,   '    ,    '   ,",
    "  '   ,    '    ,   ' ",
    "    ,    '   ,    '   ",
  ],
  [
    "  '    ,   '    ,   ' ",
    "    ,    '   ,    '   ",
    "   ,   '    ,    '   ,",
  ],
  [
    "    ,    '   ,    '   ",
    "   ,   '    ,    '   ,",
    "  '    ,   '    ,   ' ",
  ],
] as const

// heavier rain (during drag)
export const RAIN_HEAVY_FRAMES: readonly string[][] = [
  [
    "  , ' , ' , ' , ' , ' ",
    " ' , ' , ' , ' , ' ,  ",
    "  , ' , ' , ' , ' , ' ",
  ],
  [
    " ' , ' , ' , ' , ' ,  ",
    "  , ' , ' , ' , ' , ' ",
    " ' , ' , ' , ' , ' ,  ",
  ],
] as const

/* ------------------------------------------------------------------ */
/*  LIGHTNING — 3-frame zigzag bolt                                   */
/* ------------------------------------------------------------------ */

export const LIGHTNING_FRAMES: readonly string[] = [
  '          \\            ',
  '           \\\\          ',
  '            /          ',
] as const

export const LIGHTNING_FRAMES_2: readonly string[] = [
  '           \\           ',
  '          //           ',
  '          /            ',
] as const

export const LIGHTNING_FRAMES_3: readonly string[] = [
  '            \\          ',
  '           //          ',
  '            \\\\         ',
] as const

export const ALL_LIGHTNING: readonly (readonly string[])[] = [
  LIGHTNING_FRAMES,
  LIGHTNING_FRAMES_2,
  LIGHTNING_FRAMES_3,
] as const

/* ------------------------------------------------------------------ */
/*  UMBRELLA PERSON — walks along ridge, 2 walking frames             */
/* ------------------------------------------------------------------ */

export const UMBRELLA_WALK: readonly string[][] = [
  [
    '   _____  ',
    '  / ___ \\ ',
    ' /  | |  \\',
    '    |    ',
    '   / \\   ',
  ],
  [
    '   _____  ',
    '  / ___ \\ ',
    ' /  | |  \\',
    '    |    ',
    '   / |   ',
  ],
] as const

export const UMBRELLA_SPLASH: readonly string[] = [
  '. o .  ',
  '  ` `  ',
] as const

/* ------------------------------------------------------------------ */
/*  FROG — tiny 2-line frog with ribbit bubble                        */
/* ------------------------------------------------------------------ */

export const FROG_SIT: readonly string[] = [
  ' @..@ ',
  '(----)  ~',
] as const

export const FROG_HOP: readonly string[] = [
  '  @..@  ',
  ' (----) ',
] as const

export const FROG_RIBBIT: readonly string[] = [
  ' @..@ ',
  '(----) ~ribbit',
] as const

/* ------------------------------------------------------------------ */
/*  COMPOSITE HELPER: total scene height constants                    */
/* ------------------------------------------------------------------ */

export const MEGH_HEIGHT = MEGH_IDLE.length          // 5
export const RAIN_HEIGHT = RAIN_FRAMES[0].length     // 3
export const PERSON_HEIGHT = UMBRELLA_WALK[0].length  // 5
export const FROG_HEIGHT = FROG_SIT.length            // 2

export const SCENE_WIDTH = 50   // character columns for the full scene
export const SCENE_HEIGHT = MEGH_HEIGHT + RAIN_HEIGHT + 2 + PERSON_HEIGHT + FROG_HEIGHT  // ~17 lines
