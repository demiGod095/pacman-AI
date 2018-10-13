(define (domain ghost)
  (:requirements :typing :conditional-effects)
  (:types pos)
  ;; Define the problem facts
  ;; "?" denotes a variable, "-" denotes a type
  (:predicates  (At ?p - pos)
                (PacmanAt ?p - pos)
                (Adjacent ?pos1 ?pos2 - pos)
                (Scared)    ;; Whether Ghost is scared of Pacman
  )
  ;; Define the actions
  (:action move
        :parameters (?posCurr ?posNext - pos)
        :precondition (and (At ?posCurr)
                           (Adjacent ?posCurr ?posNext)
                       )
        :effect   (and (At ?posNext)
                       (not  (At ?posCurr) )
                       (when (not (Scared)) (not  (PacmanAt ?posNext) ) )
                   )
   )
)