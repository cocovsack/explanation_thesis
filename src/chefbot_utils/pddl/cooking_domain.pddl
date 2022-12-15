(define (domain cooking)
    (:requirements :typing :strips :existential-precondition :universal-precondition)
    (:types ; Defines types for diff objects
     appliance moveable robot location action_type - object
     food container tool - moveable
     pastry liquid meal ingredient - food
            )
    (:predicates

        ;; Locations objects can be in
        (inworkspace ?x - object)
        (instorage ?x - moveable)
        (incontainer ?x - food ?c - container)
        (inappliance ?x - moveable ?y - appliance)

        (ison ?a - appliance)
        (occupied ?a - appliance)
        ;; (isspreadon ?spread - topping ?toast - bread)

        (issay ?at - action_type)
        (isdo ?at - action_type)

        ;; Predicates defining various ingredients/utensils/appliance
        (ispan ?x - container)
        (iswater ?x - liquid)
        (ismilk ?x - liquid)
        (isoats ?x - ingredient)
        (issalt ?x - ingredient)
        (isbowl ?x         - container)
        (ismixingspoon ?x  - tool)
        (iseatingspoon ?x  - tool)
        ;; (iswhitebread ?x   - bread)
        ;; (isrye ?x     - bread)
        (isbanana ?x       - ingredient)
        (ischerry ?x       - ingredient)
        (isstrawberry ?x   - ingredient)
        (isblueberry ?x    - ingredient)
        (ispeanutbutter ?x - ingredient)
        ;; (isjelly ?x        - topping)
        ;; (istomato ?x       - topping)
        (ismeasuringcup ?x - container)
        (ischocolatechips ?x - ingredient)
        (isnuts ?x - ingredient)
        (ispie ?x          - pastry)
        (ismuffin ?x       - pastry)
        (isroll ?x - pastry)
        (isjellypastry ?x - pastry)
        (isegg ?x - pastry)
        (iscocopuffs ?x - ingredient)
        (iscereal ?x - meal)

        ;; For appliances
        (ismicrowave ?x - appliance)
        (istoaster ?x - appliance)
        (isstove ?x - appliance)
        (issink ?x - appliance)

        ; Not currently being used
        ;; (inhand ?x - moveable ?robot - robot)
        (inhand ?x - moveable)
        (handempty ?robot - robot)

        ;; States of ingredients/containers
        ;; (mixed ?c - container)
        (mixed ?new - meal)
        (heated ?x - moveable)
        ;; (toasted ?x - bread)
        (warmed ?x - pastry)
        (boiling ?x - food)
        (simmering ?x - meal)

        (hypothetical ?new - meal)
        (error ?new - meal)
        (hypotheticalpastry ?new - meal)

        (isoatmeal ?new - meal)
        (isplainoatmeal ?new - meal)
        (isfruityoatmeal ?new - meal)
        (isnuttyoatmeal ?new - meal)
        (ischocolateoatmeal ?new - meal)
        (isfruitychocolateoatmeal ?new - meal)
        (ispeanutbutterbananaoatmeal ?new - meal)
        (ispeanutbutterchocolateoatmeal ?new - meal)
        (isnuttyfruityoatmeal ?new - meal)
        (isnuttychocolateoatmeal ?new - meal)
        (isnuttyfruitychocolateoatmeal ?new - meal)
        (isnuttypeanutbutterbananaoatmeal ?new - meal)
        (isnuttypeanutbutterchocolateoatmeal ?new - meal)

        ;; (ispbbananatoast ?new - meal)
        (iscompletedoatmeal ?new - meal)
        (iscompletedcereal ?new - meal)
        ;; (iscompletedtoast ?new - meal)
        (iscompletedpastry ?new - meal)

        (iscompletedmuffin ?new - meal)
        (iscompletedpie ?new - meal)
        (iscompletedjellypastry ?new - meal)
        (iscompletedegg ?new - meal)
        (iscompletedroll ?new - meal)

        ;; Action literals
        ;; (mix ?w - liquid ?c - container ?new - meal ?at - action_type)
        (mix ?c - container ?new - meal ?at - action_type)
        (gather ?x - moveable ?at - action_type)
        (pour ?x - food ?c - container ?at - action_type)
        (pourwater ?dest - container ?at - action_type)
        ;; (grabspoon ?t - tool ?at - action_type)
        (turnon ?a - appliance ?at - action_type)
        (wait ?a - appliance ?at - action_type)

        
        (errormultipleliquidsmixed ?new - meal)
        (erroraddingredientstooearly ?l - liquid ?i - ingredient ?new - meal)

        (checkcereal ?new - meal ?at - action_type)

        (checkplainoatmeal ?c - container ?new - meal  ?at - action_type)
        (checkfruityoatmeal ?c - container ?new - meal ?at - action_type)
        (checknuttyoatmeal ?c - container ?new - meal ?at - action_type)
        (checkchocolateoatmeal ?c - container ?new - meal ?at - action_type)
        (checkfruitychocolateoatmeal ?c - container ?new - meal ?at - action_type)
        (checkpeanutbutterbananaoatmeal ?c - container ?new - meal ?at - action_type)
        (checkpeanutbutterchocolateoatmeal ?c - container ?new - meal ?at - action_type)
        (checknuttyfruityoatmeal ?c - container ?new - meal ?at - action_type)
        (checknuttychocolateoatmeal ?c - container ?new - meal ?at - action_type)
        (checknuttyfruitychocolateoatmeal ?c - container ?new - meal ?at - action_type)
        (checknuttypeanutbutterbananaoatmeal ?c - container ?new - meal ?at - action_type)
        (checknuttypeanutbutterchocolateoatmeal ?c - container ?new - meal ?at - action_type)

        (putin ?x - container ?a - appliance ?at - action_type)
        (putinsink ?mc - container ?sink - appliance ?at - action_type)
        (putinmicrowave ?x - pastry  ?at - action_type)
        ;; (toastbread ?x - ingredient)
        ;; (spread ?toast - bread ?spread - topping)
        ;; (checkpbbananatoast ?new - meal)
        ;; (checkjellytoast ?new - meal)
        ;; (checktomatotoast ?new - meal)
        (microwavepastry ?x - pastry ?at - action_type)
        (takeoutmicrowave ?x - pastry  ?at - action_type)
        (checkpie ?new - meal ?at - action_type)
        (checkmuffin  ?new - meal ?at - action_type)
        (checkroll   ?new - meal ?at - action_type)
        (checkjellypastry   ?new - meal ?at - action_type)
        (checkegg   ?new - meal ?at - action_type)
        
        ;; (releasetool ?t - tool)
        (collectwater ?cup - container ?water - liquid ?at - action_type)
        ;; (boilliquid ?pan - container ?w - liquid ?at - action_type)
        (boilliquid ?new - meal ?pan - container ?at - action_type)
        (reduceheat ?pan - container ?new - meal ?at - action_type)
        (cookoatmeal ?stove - appliance ?new - meal ?at - action_type)
        (serveoatmeal ?pan - container ?bowl - container ?new - meal ?at - action_type)

        ); (:actions gather pour pourwater mix takeoutmicrowave checkplainoatmeal turnon wait putinmicrowave  microwavepastry checkpie checkmuffin checkroll checkegg checkjellypastry  collectwater boilliquid reduceheat cookoatmeal checkfruityoatmeal checknuttyoatmeal checkchocolateoatmeal checkfruitychocolateoatmeal checkpeanutbutterbananaoatmeal checkpeanutbutterchocolateoatmeal checknuttyfruityoatmeal checknuttychocolateoatmeal checknuttyfruitychocolateoatmeal checknuttypeanutbutterbananaoatmeal checknuttypeanutbutterchocolateoatmeal serveoatmeal checkcereal errormultipleliquidsmixed)

    ; Bring ingredients from storage to workspace
    (:action gather
             :parameters (?item - moveable ?robot - robot ?at - action_type)
             :precondition (and
                            ;; (not (iswater ?item))
                            (gather ?item ?at)
                            (isdo ?at)
                            ;; (handempty ?robot)
                            (not (inworkspace ?item))
                            (instorage ?item)
                            )
             :effect (and
                      (not (instorage ?item))
                      (inworkspace ?item)
                      )

             )

    ; Put object inside an appliance
    (:action put_in
             :parameters (?x - container ?a - appliance ?r - robot ?at - action_type)
             :precondition (and
                            (putin ?x ?a ?at)
                            (inworkspace ?x) ; Do we need ingredients to be in workspace first?
                            (not (ismicrowave ?a))
                            ;; (handempty ?r)
                            (not (inappliance ?x ?a))
                            (not (occupied ?a))
                            )
             :effect (and
                      (inappliance ?x ?a)
                      ;; (not(inworkspace ?x)) ; TODO: Is this needed?
                      (occupied ?a)
                      )
     )


    (:action put_in_sink
             :parameters (?x - container ?a - appliance ?r - robot ?at - action_type)
             :precondition (and
                            (putinsink ?x ?a ?at)
                            (issink ?a)
                            (ismeasuringcup ?x)
                            ;; (handempty ?r)
                            (not (inappliance ?x ?a))
                            (not (occupied ?a))
                            )
             :effect (and
                      (inappliance ?x ?a)
                      ;; (not(inworkspace ?x)) ; TODO: Is this needed?
                      (occupied ?a)
                      )
             )

    (:action put_in_microwave
             :parameters (?x - pastry ?a - appliance ?r - robot ?at - action_type)
             :precondition (and
                            (putinmicrowave ?x ?at)
                            (inworkspace ?x) ; Do we need ingredients to be in workspace first?
                            (ismicrowave ?a)
                            ;; (handempty ?r)
                            (not (inappliance ?x ?a))
                            (not (occupied ?a))
                            ;; (not (ison ?a))
                            )
             :effect (and
                      (inappliance ?x ?a)
                      (not (ison ?a))
                      ;; (not(inworkspace ?x)) ; TODO: Is this needed?
                      (occupied ?a)
                      )
             )


    ; Remove item from microwave
    (:action take_out_microwave
             :parameters (?x - pastry ?a - appliance ?r - robot ?at - action_type)
             :precondition (and
                            ;; (handempty ?r)
                            (takeoutmicrowave ?x ?at)
                            (ismicrowave ?a)
                            (inappliance ?x ?a)
                            ;; (not (ison ?a))
                            ;; (warmed ?x)


                            )
             :effect (and
                      (not (occupied ?a))
                      (not (inappliance ?x ?a))
                      (inworkspace ?x)
                      )
             )



    ; put ingredients into a container
    ; NOTE: Currently does  not deal with the case of transferring from one container
    ; to another e.g. pouring water in pan. Maybe need seperate action for that?
     (:action pour
              :parameters (?x - food ?c - container ?robot - robot ?at - action_type)
              :precondition (and
                             (pour ?x ?c ?at)
                             ;; (handempty ?robot)
                             ;; (not (iswater ?x))
                             (not (ismeasuringcup ?c))
                             (not (incontainer ?x ?c))
                             (not (isjellypastry ?x))
                             (not (isegg ?x))
                             (not (isroll ?x))
                             (not (ismuffin ?x))
                             (not (ispie ?x))
                             (inworkspace ?x)
                             (inworkspace ?c)
                             )
              :effect (and
                       (incontainer ?x ?c)
                       (not (inworkspace ?x))
                       )
              )
     (:action pour_water
              :parameters (?w - liquid ?c - container ?dest - container ?at - action_type)
              :precondition (and
                             (pourwater ?dest ?at)
                             ;; (handempty ?robot)
                             (iswater ?w)
                             (ismeasuringcup ?c)
                             (incontainer ?w ?c)
                             (inworkspace ?dest)
                             (inworkspace ?c)
                             )
              :effect (and
                       (incontainer ?w ?dest)
                       (not (incontainer ?w ?c))
                       ;; (not (inworkspace ?x))
                       )
              )

     ; Mix oatmeal ingredients in a pot
     (:action mix
              :parameters (?pan - container ?w - liquid ?o - ingredient ?s - ingredient  ?tool - tool ?a - appliance  ?new - meal ?at - action_type)
              :precondition (and
                              (mix ?pan ?new ?at)
                              (hypothetical ?new)
                              ;; (iswater ?w)
                              (boiling ?new)
                              (issalt ?s)
                              (isoats ?o)
                              (isstove ?a)
                              (ispan ?pan)
                              (ismixingspoon ?tool)
                              (incontainer ?w ?pan)
                              (incontainer ?s ?pan)
                              (incontainer ?o ?pan)
                              (ison ?a)
                              (inappliance ?pan ?a)
                              (not (mixed ?new))
                              ;; (inhand ?tool ?robot)
                              ;; (inworkspace ?tool)
                              ;; (not (mixed ?pan))
                              )

              :effect (and
                     (not (issalt ?s))
                       (not (isoats ?o))
                       ;; (not (iswater ?w)) ; TODO: remove this
                       ;; (not (boiling ?w))
                       (not (incontainer ?w ?pan))
                       (not (incontainer ?s ?pan))
                       (not (incontainer ?o ?pan))
                       ;; (not (boiling ?w))
                       ;; (not (inhand ?tool ?robot))
                       (not (inhand ?tool))
                       (instorage ?tool)
                       ;; (not (hypothetical ?new))
                       (mixed ?new)
                       ;; (boiling ?new)
                       (incontainer ?new ?pan)
                       )
              )



     ; Turn the heat down on stove to simmer oatmeal
     (:action reduce_heat
              :parameters (?pan - container ?new - meal ?stove - appliance ?at - action_type)
              :precondition(and
                            (reduceheat ?pan ?new ?at)
                            (ison ?stove)
                            (isstove ?stove)
                            (ispan ?pan)
                            (boiling ?new)
                            (mixed ?new)
                            (inappliance ?pan ?stove)
                            (incontainer ?new ?pan)
                            )
              :effect (and
                       (not (boiling ?new))
                       (simmering ?new)
                       )

      )

     ; Wait for oatmeal to cook. Produces completed oatmeal
     (:action cook_oatmeal
              :parameters (?pan - container  ?stove - appliance ?new - meal ?at - action_type)
              :precondition(and
                            (cookoatmeal ?stove ?new ?at)
                            (issay ?at)
                            (simmering ?new)
                            (ispan ?pan)
                            (isstove ?stove)
                            (ison ?stove)
                            (inappliance ?pan ?stove)
                            (hypothetical ?new)
                            (incontainer ?new ?pan)
                            )
              :effect(and
                      (not (simmering ?new))
                      (not (hypothetical ?new))
                      (isoatmeal ?new)
                      )
      )


     ; Wait for oatmeal to cook. Produces completed oatmeal
     (:action check_cereal
              :parameters (?bowl - container ?m - liquid ?c - ingredient ?spoon - tool ?new - meal ?at - action_type)
              :precondition(and
                            (checkcereal ?new ?at)
                            (not (iscompletedcereal ?new))
                            (issay ?at)
                            (hypothetical ?new)
                            (iscocopuffs ?c)
                            (ismilk ?m)
                            (isbowl ?bowl)
                            (iseatingspoon ?spoon)
                            (incontainer ?c ?bowl)
                            (incontainer ?m ?bowl)
                            (inworkspace ?spoon)
                            )
              :effect(and
                      (not (iscocopuffs ?c))
                      (not (ismilk ?m))
                      (not (hypothetical ?new))
                      (iscereal ?new)
                      (incontainer ?new ?bowl)
                      ;; (iscompletedoatmeal ?new)
                      (iscompletedcereal ?new)
                      )
              )




     ; Transfer oatmeal from pan to bowl
     (:action serve_oatmeal
              :parameters (?pan - container ?new - meal ?bowl - container ?at - action_type)
              :precondition(and
                            (serveoatmeal ?pan ?bowl  ?new ?at)
                            (isoatmeal ?new)
                            (ispan ?pan)
                            (isbowl ?bowl)
                            (incontainer ?new ?pan)
                            (not (incontainer ?new ?bowl))
                            (inworkspace ?bowl)
                            )
              :effect (and
                       (not (incontainer ?new ?pan))
                       (incontainer ?new ?bowl)
                       )
              )




     ; Is plain oatmeal done?
     (:action check_plain_oatmeal
              :parameters (?new - meal ?t - tool ?bowl - container ?at - action_type)
              :precondition (and
                             (checkplainoatmeal ?bowl ?new ?at)
                             (not (isplainoatmeal ?new))
                             (issay ?at)
                             (isbowl ?bowl)
                             (iseatingspoon ?t)
                             (inworkspace ?t)
                             (incontainer ?new ?bowl)
                             (isoatmeal ?new)
                             )
              :effect (and

                       (iscompletedoatmeal ?new)
                       (isplainoatmeal ?new)
                       )
              )




    ; Is plain fruity done?
     (:action check_fruity_oatmeal
              :parameters (?new - meal ?t - tool ?b - ingredient ?bb - ingredient ?sb - ingredient ?bowl - container ?at - action_type)
              :precondition (and
                             (checkfruityoatmeal ?bowl ?new ?at)
                             (not (isfruityoatmeal ?new))
                             (issay ?at)
                             (isbanana ?b)
                             (isblueberry ?bb)
                             (isstrawberry ?sb)
                             (isbowl ?bowl)
                             (isoatmeal ?new)
                             (iseatingspoon ?t)
                             (inworkspace ?t)
                             (incontainer ?new ?bowl)
                             (incontainer ?bb ?bowl)
                             (incontainer ?b ?bowl)
                             (incontainer ?sb ?bowl)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (isfruityoatmeal ?new)
               (not (incontainer ?b ?bowl))
               (not (incontainer ?bb ?bowl))
               (not (incontainer ?sb ?bowl))
               )

              )

    ; Is plain fruity done?
     (:action check_fruity_chocolate_oatmeal
              :parameters (?new - meal ?t - tool ?b - ingredient ?bb - ingredient ?sb - ingredient ?cc - ingredient   ?bowl - container ?at - action_type)
              :precondition (and
                             (checkfruitychocolateoatmeal ?bowl ?new ?at)
                             (not (isfruitychocolateoatmeal ?new))
                             (issay ?at)
                             (isbanana ?b)
                             (isblueberry ?bb)
                             (isstrawberry ?sb)
                             (ischocolatechips ?cc)
                             (isbowl ?bowl)
                             (isoatmeal ?new)
                             (iseatingspoon ?t)
                             (inworkspace ?t)
                             (incontainer ?new ?bowl)
                             (incontainer ?bb ?bowl)
                             (incontainer ?b ?bowl)
                             (incontainer ?sb ?bowl)
                             (incontainer ?cc ?bowl)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (isfruitychocolateoatmeal ?new)
               (not (incontainer ?b ?bowl))
               (not (incontainer ?bb ?bowl))
               (not (incontainer ?sb ?bowl))
               (not (incontainer ?cc ?bowl))
               )

              )

    ; Is plain fruity done?
     (:action check_nutty_fruity_oatmeal
              :parameters (?new - meal ?t - tool ?b - ingredient ?bb - ingredient ?sb - ingredient ?n - ingredient   ?bowl - container ?at - action_type)
              :precondition (and
                             (checknuttyfruityoatmeal ?bowl ?new ?at)
                             (not (isnuttyfruityoatmeal ?new))
                             (issay ?at)
                             (isbanana ?b)
                             (isblueberry ?bb)
                             (isstrawberry ?sb)
                             (isnuts ?n)
                             (isbowl ?bowl)
                             (isoatmeal ?new)
                             (iseatingspoon ?t)
                             (inworkspace ?t)
                             (incontainer ?new ?bowl)
                             (incontainer ?bb ?bowl)
                             (incontainer ?b ?bowl)
                             (incontainer ?sb ?bowl)
                             (incontainer ?n ?bowl)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (isnuttyfruityoatmeal ?new)
               (not (incontainer ?b ?bowl))
               (not (incontainer ?bb ?bowl))
               (not (incontainer ?sb ?bowl))
               (not (incontainer ?n ?bowl))
               )

              )



    ; Is fruity chocolate oateal done?
     (:action check_fruity_chocolate_oatmeal
              :parameters (?new - meal ?t - tool ?b - ingredient ?bb - ingredient ?sb - ingredient ?cc - ingredient   ?bowl - container ?at - action_type)
              :precondition (and
                             (checkfruitychocolateoatmeal ?bowl ?new ?at)
                             (not (isfruitychocolateoatmeal ?new))
                             (issay ?at)
                             (isbanana ?b)
                             (isblueberry ?bb)
                             (isstrawberry ?sb)
                             (ischocolatechips ?cc)
                             (isbowl ?bowl)
                             (isoatmeal ?new)
                             (iseatingspoon ?t)
                             (inworkspace ?t)
                             (incontainer ?new ?bowl)
                             (incontainer ?bb ?bowl)
                             (incontainer ?b ?bowl)
                             (incontainer ?sb ?bowl)
                             (incontainer ?cc ?bowl)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (isfruitychocolateoatmeal ?new)
               (not (incontainer ?b ?bowl))
               (not (incontainer ?bb ?bowl))
               (not (incontainer ?sb ?bowl))
               (not (incontainer ?cc ?bowl))
               ))


    ; Is nutty oatmeal done?
     (:action check_nutty_oatmeal
              :parameters (?new - meal ?t- tool ?n - ingredient  ?bowl - container ?at - action_type)
              :precondition (and
                             (checknuttyoatmeal ?bowl ?new ?at)
                             (not (isnuttyoatmeal ?new))
                             (issay ?at)
                             (isnuts ?n)
                             (isbowl ?bowl)
                             (isoatmeal ?new)
                             (iseatingspoon ?t)
                             (inworkspace ?t)
                             (incontainer ?new ?bowl)
                             (incontainer ?n ?bowl)
                             ;; (incontainer ?b ?bowl)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (isnuttyoatmeal ?new)
               (not (incontainer ?n ?bowl))
               )

              )



    ; Is chocolate oatmeal done?
     (:action check_chocolate_oatmeal
              :parameters (?new - meal ?t - tool ?c - ingredient  ?bowl - container ?at - action_type)
              :precondition (and
                             (checkchocolateoatmeal ?bowl ?new ?at)
                             (not (ischocolateoatmeal ?new))
                             (issay ?at)
                             (ischocolatechips ?c)
                             (isbowl ?bowl)
                             (isoatmeal ?new)
                             (iseatingspoon ?t)
                             (inworkspace ?t)
                             (incontainer ?new ?bowl)
                             (incontainer ?c ?bowl)
                             ;; (incontainer ?b ?bowl)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (ischocolateoatmeal ?new)
               (not (incontainer ?c ?bowl))
               )

              )

     ; Is chocolate oatmeal done?
     (:action check_peanutbutter_banana_oatmeal
              :parameters (?new - meal ?t - tool ?pb - ingredient ?b - ingredient  ?bowl - container ?at - action_type)
              :precondition (and
                             (checkpeanutbutterbananaoatmeal ?bowl ?new ?at)
                             (not (ispeanutbutterbananaoatmeal ?new))
                             (issay ?at)
                             (ispeanutbutter ?pb)
                             (isbanana ?b)
                             (isbowl ?bowl)
                             (isoatmeal ?new)
                             (iseatingspoon ?t)
                             (inworkspace ?t)
                             (incontainer ?new ?bowl)
                             (incontainer ?pb ?bowl)
                             (incontainer ?b ?bowl)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (ispeanutbutterbananaoatmeal ?new)
               (not (incontainer ?pb ?bowl))
               (not (incontainer ?b ?bowl))
               ))


     ; Is chocolate oatmeal done?
     (:action check_peanutbutter_chocolate_oatmeal
              :parameters (?new - meal ?t - tool ?pb - ingredient ?c - ingredient  ?bowl - container ?at - action_type)
              :precondition (and
                             (checkpeanutbutterchocolateoatmeal ?bowl ?new ?at)
                             (not (ispeanutbutterchocolateoatmeal ?new))
                             (issay ?at)
                             (ispeanutbutter ?pb)
                             (ischocolatechips ?c)
                             (isbowl ?bowl)
                             (isoatmeal ?new)
                             (iseatingspoon ?t)
                             (inworkspace ?t)
                             (incontainer ?new ?bowl)
                             (incontainer ?pb ?bowl)
                             (incontainer ?c ?bowl)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (ispeanutbutterchocolateoatmeal ?new)
               (not (incontainer ?pb ?bowl))
               (not (incontainer ?c ?bowl))
               )

              )

     ; Is chocolate oatmeal done?
     (:action check_nutty_chocolate_oatmeal
              :parameters (?new - meal ?t - tool ?cc - ingredient ?n - ingredient  ?bowl - container ?at - action_type)
              :precondition (and
                             (checknuttychocolateoatmeal ?bowl ?new ?at)
                             (not (isnuttychocolateoatmeal ?new))
                             (issay ?at)
                             (ischocolatechips ?cc)
                             (isnuts ?n)
                             (isbowl ?bowl)
                             (isoatmeal ?new)
                             (iseatingspoon ?t)
                             (inworkspace ?t)
                             (incontainer ?new ?bowl)
                             (incontainer ?cc ?bowl)
                             (incontainer ?n ?bowl)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (isnuttychocolateoatmeal ?new)
               (not (incontainer ?cc ?bowl))
               (not (incontainer ?n ?bowl))
               )
              )

    ; Is plain fruity done?
     (:action check_nutty_peanutbutter_banana_oatmeal
              :parameters (?new - meal ?t - tool ?b - ingredient ?n - ingredient ?pb - ingredient ?bowl - container ?at - action_type)
              :precondition (and
                             (checknuttypeanutbutterbananaoatmeal ?bowl ?new ?at)
                             (not (isnuttypeanutbutterbananaoatmeal ?new))
                             (issay ?at)
                             (isbanana ?b)
                             (isnuts ?n)
                             (ispeanutbutter ?pb)
                             (isbowl ?bowl)
                             (isoatmeal ?new)
                             (iseatingspoon ?t)
                             (incontainer ?new ?bowl)
                             (incontainer ?n ?bowl)
                             (incontainer ?b ?bowl)
                             (incontainer ?pb ?bowl)
                             (inworkspace ?t)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (isnuttypeanutbutterbananaoatmeal ?new)
               (not (incontainer ?b ?bowl))
               (not (incontainer ?n ?bowl))
               (not (incontainer ?pb ?bowl))
               ))

     (:action check_nutty_peanutbutter_chocolate_oatmeal
              :parameters (?new - meal ?t - tool ?n - ingredient ?pb - ingredient ?cc - ingredient   ?bowl - container ?at - action_type)
              :precondition (and
                             (checknuttypeanutbutterchocolateoatmeal ?bowl ?new ?at)
                             (not (isnuttypeanutbutterchocolateoatmeal ?new))
                             (issay ?at)
                             (isnuts ?n)
                             (ispeanutbutter ?pb)
                             (ischocolatechips ?cc)
                             (isbowl ?bowl)
                             (isoatmeal ?new)
                             (iseatingspoon ?t)
                             (incontainer ?new ?bowl)
                             (incontainer ?n ?bowl)
                             (incontainer ?pb ?bowl)
                             (incontainer ?cc ?bowl)
                             (inworkspace ?t)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (isnuttypeanutbutterchocolateoatmeal ?new)
               (not (incontainer ?n ?bowl))
               (not (incontainer ?pb ?bowl))
               (not (incontainer ?cc ?bowl))
               ))


     (:action check_nutty_fruity_chocolate_oatmeal
              :parameters (?new - meal ?t - tool ?b - ingredient ?bb - ingredient ?sb - ingredient ?cc - ingredient  ?n - ingredient  ?bowl - container ?at - action_type)
              :precondition (and
                             (checknuttyfruitychocolateoatmeal ?bowl ?new ?at)
                             (not (isnuttyfruitychocolateoatmeal ?new))
                             (issay ?at)
                             (isbanana ?b)
                             (isnuts ?n)
                             (isblueberry ?bb)
                             (isstrawberry ?sb)
                             (ischocolatechips ?cc)
                             (isbowl ?bowl)
                             (iseatingspoon ?t)
                             (isoatmeal ?new)
                             (incontainer ?new ?bowl)
                             (incontainer ?bb ?bowl)
                             (incontainer ?b ?bowl)
                             (incontainer ?sb ?bowl)
                             (incontainer ?cc ?bowl)
                             (incontainer ?n ?bowl)
                             (inworkspace ?t)
                             )
              :effect
              (and
               (iscompletedoatmeal ?new)
               (isnuttyfruitychocolateoatmeal ?new)
               (not (incontainer ?b ?bowl))
               (not (incontainer ?bb ?bowl))
               (not (incontainer ?sb ?bowl))
               (not (incontainer ?cc ?bowl))
               (not (incontainer ?n ?bowl))
               ))





     ;; (:action grab_spoon
     ;;          :parameters (?t - tool ?at - action_type)
     ;;          :precondition (and
     ;;                         (grabspoon ?t ?at)
     ;;                         (isdo ?at)
     ;;                         (instorage ?t)
     ;;                         (ismixingspoon ?t)
     ;;                         ;; (handempty ?r)
     ;;                         ;; (not (inhand ?t ?r))
     ;;                         (not (inhand ?t))
     ;;                         (not (inworkspace ?t))
     ;;                         ;; (instorage ?t)
     ;;                         )
     ;;          :effect (and
     ;;                   ;; (inhand ?t ?r)
     ;;                   (inhand ?t)
     ;;                   (not (instorage ?t))
     ;;                   ;; (not (inworkspace ?t))
     ;;                   (inworkspace ?t)
     ;;                   ;; (not (handempty ?r))
     ;;                   )
     ;;          )


     ; Get water from sink. Fills a container with water.
     (:action collect_water
              :parameters (?c - container  ?sink - appliance ?water - liquid ?at - action_type)
              :precondition (and
                             (collectwater ?c ?water ?at)
                             (issink ?sink)
                             ;; (inappliance ?c ?sink)
                             (ismeasuringcup ?c)
                             (inworkspace ?c)
                             (inappliance ?water ?sink )
                             ;; (hypothetical ?new)
                             )
              :effect (and
                       ;; (iswater ?new)
                       ;; (inworkspace ?new)
                       (incontainer ?water ?c)
                       (not (inappliance ?water ?sink))
                       ;; (not (hypothetical ?new))
                       )
              )



     ; Boil water in pan
     (:action boil_liquid
              :parameters (?new - meal ?pan - container ?w - liquid ?stove - appliance ?at - action_type)
              :precondition (and
                             (boilliquid ?new ?pan  ?at)
                             (hypothetical ?new)
                             (issay ?at)
                             (isstove ?stove)
                             (ispan ?pan)
                             ;; (iswater ?w)
                             (incontainer ?w ?pan)
                             (inappliance ?pan ?stove)
                             (ison ?stove)
                             )
              :effect (and (boiling ?new))
              )

     ;; (:action release_tool
     ;;          :parameters (?t - tool ?r - robot)
     ;;          :precondition (and
     ;;                         (releasetool ?t)
     ;;                         (ismixingspoon ?t)
     ;;                         (not (handempty ?r))
     ;;                         (inhand ?t ?r)
     ;;                         )
     ;;          :effect (and
     ;;                   (handempty ?r)
     ;;                   (not (inhand ?t ?r))
     ;;                   )
     ;;          )

     ; Turn on an appliance
     (:action turn_on
              :parameters (?a - appliance  ?r - robot ?at - action_type)
              :precondition (and
                             (turnon ?a ?at)
                             (not(issink ?a))
                             ;; (handempty ?r)
                             (not (ison ?a))
                             ;; (inappliance ?x ?a)

                             )
              :effect (and (ison ?a)
                           ;; (heated ?x)
                           )

              )



    ; Wait for microwave to complete heating a pastry. 
    (:action microwave_pastry
             :parameters (?p - pastry ?a - appliance ?at - action_type)
             :precondition (and
                            (microwavepastry ?p ?at)
                            (issay ?at)
                            (ismicrowave ?a)
                            (ison ?a)
                            (inappliance ?p ?a)
                            )

             :effect (and
                      (warmed ?p)
                      (not (ison ?a))
                      )
             )



    ; Is pie finished?
     (:action check_pie
              :parameters (?p - pastry ?new - meal ?a - appliance ?at - action_type)
              :precondition (and
                             (checkpie ?new ?at)
                             (issay ?at)
                             (ispie ?p)
                             (warmed ?p)
                             (ismicrowave ?a)
                             (not (inappliance ?p ?a))
                             (inworkspace ?p)
                             (hypotheticalpastry ?new)
                             )
              :effect (and
                       (iscompletedpastry ?new)
                       (iscompletedpie ?new)
                       (not (inworkspace ?p))
                       (not (ispie ?p))
                       (not (hypotheticalpastry ?new))
                       )
              )

     ; is muffin completed
     (:action check_muffin
              :parameters (?p - pastry ?new - meal ?a - appliance ?at - action_type)
              :precondition (and
                             (checkmuffin ?new ?at)
                             (issay ?at)
                             (warmed ?p)
                             (ismuffin ?p)
                             (ismicrowave ?a)
                             (not (inappliance ?p ?a))
                             (inworkspace ?p)
                             (hypotheticalpastry ?new)
                             )
              :effect (and
                       (iscompletedpastry ?new)
                       (iscompletedmuffin ?new)
                       (not (inworkspace ?p))
                       (not (ismuffin ?p))
                       (not (hypotheticalpastry ?new))
                       )
              )

                                        ; is muffin completed
     (:action check_roll
              :parameters (?p - pastry ?new - meal ?a - appliance ?at - action_type)
              :precondition (and
                             (checkroll ?new ?at)
                             (issay ?at)
                             (warmed ?p)
                             (isroll ?p)
                             (ismicrowave ?a)
                             (not (inappliance ?p ?a))
                             (inworkspace ?p)
                             (hypotheticalpastry ?new)
                             )
              :effect (and
                       (iscompletedpastry ?new)
                       (iscompletedroll ?new)
                       (not (inworkspace ?p))
                       (not (isroll ?p))
                       (not (hypothetical ?new))
                       )
              )

     (:action check_jelly_pastry
              :parameters (?p - pastry ?new - meal ?a - appliance ?at - action_type)
              :precondition (and
                             (checkjellypastry ?new ?at)
                             (issay ?at)
                             (warmed ?p)
                             (isjellypastry ?p)
                             (ismicrowave ?a)
                             (not (inappliance ?p ?a))
                             (inworkspace ?p)
                             (hypotheticalpastry ?new)
                             )
              :effect (and
                       (iscompletedpastry ?new)
                       (iscompletedjellypastry ?new)
                       (not (inworkspace ?p))
                       (not (isjellypastry ?p))
                       (not (hypotheticalpastry ?new))
                       )
              )


     (:action check_egg
              :parameters (?p - pastry ?new - meal ?a - appliance ?at - action_type)
              :precondition (and
                             (checkegg ?new ?at)
                             (issay ?at)
                             (warmed ?p)
                             (isegg ?p)
                             (ismicrowave ?a)
                             (not (inappliance ?p ?a))
                             (inworkspace ?p)
                             (hypotheticalpastry ?new)
                             )
              :effect (and
                       (iscompletedpastry ?new)
                       (iscompletedegg ?new)
                       (not (inworkspace ?p))
                       (not (isegg ?p))
                       (not (hypotheticalpastry ?new))
                       )
              )

     (:action error_multiple_liquids_mixed
              :parameters (?m - liquid ?w - liquid ?p - container ?new - meal)
              :precondition (and
                             (errormultipleliquidsmixed ?new)
                             (ismilk ?m)
                             (iswater ?w)
                             (hypothetical ?new)
                             (ispan ?p)
                             (incontainer ?m ?p)
                             (incontainer ?w ?p)
                             )
              :effect (and
                       (error ?new)
                       )
              )

      (:action error_add_ingredients_too_early
               :parameters (?l - liquid ?i - ingredient ?p - container ?new - meal)
               :precondition (and
                              (erroraddingredientstooearly ?l ?i ?new)
                              (hypothetical ?new)
                              (ispan ?p)
                              (not (issalt ?i))
                              (not (boiling ?l))
                              ;; (not (simmering ?l))
                              (incontainer ?l ?p)
                              (incontainer ?i ?p)
                              )
               :effect (and
                        (error ?new)
                        )
               )


    )
