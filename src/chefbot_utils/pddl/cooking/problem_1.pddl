
(define (problem cooking)
    (:domain cooking)
    (:objects 
     robot - robot
     microwave - appliance
     stove - appliance
     sink - appliance
     ; Oatmeal ingredients
     peanutbutter - ingredient
     ;; nuts - ingredient
     water - liquid
     milk - liquid
     oats - ingredient
     salt - ingredient
     blueberry - ingredient
     strawberry - ingredient
     banana - ingredient
     chocolatechips - ingredient

     cocopuffs - ingredient

     say - action_type
     do - action_type

     pie - pastry
     muffin - pastry
     roll - pastry
     jellypastry - pastry
     egg - pastry

     bowl - container
     pan - container
     measuringcup - container
     mixingspoon - tool
     eatingspoon - tool

     main - meal
     side - meal
     ;; new3 - meal

    )
    (:init
     (handempty robot)

     (issay say)
     (isdo do)

     (isoats oats)
     (ismeasuringcup measuringcup)
     (issalt salt)

     (ispan pan)
     (isbowl bowl)
     (ismixingspoon mixingspoon)
     (iseatingspoon eatingspoon)

     ;; (isspoon mixingspoon)
     (ispie pie)
     (ismuffin muffin)
     (isroll roll)
     (isjellypastry jellypastry)
     (isegg egg)

     (ispeanutbutter peanutbutter)
     (isbanana banana)
     (isstrawberry strawberry)
     (isblueberry blueberry)
     (ismilk milk)
     ;; (isnuts nuts)
     (ischocolatechips chocolatechips)

     (iscocopuffs cocopuffs)

     (isstove stove)
     (ismicrowave microwave)
     (issink sink)
     (iswater water)
     (inappliance water sink)

     (hypothetical main)
     (hypotheticalpastry  side)
     ;; (hypothetical new3)
     ;; (hypothetical water)

     (inworkspace microwave)
     (inworkspace stove)
     (inworkspace sink)

     ;; (instorage pan)
     (inworkspace pan)
     (instorage salt)
     ;; (inworkspace mixingspoon)
     (inappliance pan stove)

     (instorage mixingspoon)
     (instorage eatingspoon)
     (instorage bowl)
     ;; (instorage mixingspoon)
     (instorage oats)
     ;; (instorage salt)
     (instorage peanutbutter)
     (instorage pie)
     (instorage muffin)
     (instorage measuringcup)
     (instorage blueberry)
     (instorage cocopuffs)
     ;; (instorage nuts)
     (instorage chocolatechips)
     (instorage strawberry)
     (instorage banana)
     (instorage milk)
     (instorage roll)
     (instorage jellypastry)
     (instorage egg)


     ;action literals
     ;; (gather oats)
     ;; (gather salt)
     ;; (gather pan)
     ;; (gather bowl)
     ;; (gather pie)
     ;; (gather milk)
     ;; (gather nuts)
     ;; (gather strawberry)
     ;; (gather blueberry)
     ;; (gather banana)
     ;; (gather peanutbutter)
     ;; (gather chocolatechips)
     ;; (gather roll)
     ;; (gather jellypastry)
     ;; (gather egg)
     
     ;; (pour main bowl)
     ;; (pour  side bowl)
     ;; (pour new3 bowl)
     ;; (pour main pan)
     ;; (pour  side pan)
     ;; (pour new3 pan)
     ;; (pour milk pan)
     ;; (pour oats pan)
     ;; (pour salt pan)
     ;; (pour nuts pan)
     ;; (pour strawberry pan)
     ;; (pour blueberry pan)
     ;; (pour banana pan)
     ;; (pour chocolatechips pan)
     ;; (pour milk bowl)
     ;; (pour oats bowl)
     ;; (pour salt bowl)
     ;; (pour nuts bowl)
     ;; (pour strawberry bowl)
     ;; (pour blueberry bowl)
     ;; (pour banana bowl)
     ;; (pour chocolatechips bowl)


     ;; (mix water pan main)
     ;; (mix water pan  side)
     ;; (mix water pan new3)
     ;; (mix water bowl main)
     ;; (mix water bowl  side)
     ;; (mix water bowl new3)
     ;; (mix milk pan main)
     ;; (mix milk pan  side)
     ;; (mix milk pan new3)
     ;; (mix milk bowl main)
     ;; (mix milk bowl  side)
     ;; (mix milk bowl new3)

     ;; (mix water pan main do)
     ;; (mix water pan  side do)
     ;; (mix milk pan main do)
     ;; (mix milk pan  side do)
     ;; (grabtool mixingspoon)

     ;; (checkplainoatmeal pan main)
     ;; (checkplainoatmeal pan  side)
     ;; (checkplainoatmeal pan new3)
     ;; (checkplainoatmeal bowl main)
     ;; (checkplainoatmeal bowl  side)
     ;; (checkplainoatmeal bowl new3)
     ;; (checkfruityoatmeal bowl main)
     ;; (checkfruityoatmeal bowl  side)
     ;; (checkfruityoatmeal bowl new3)
     ;; (checkfruityoatmeal pan main)
     ;; (checkfruityoatmeal pan  side)
     ;; (checkfruityoatmeal pan new3)
     ;; (checkchocolateoatmeal bowl main)
     ;; (checkchocolateoatmeal bowl  side)
     ;; (checkchocolateoatmeal bowl new3)
     ;; (checkchocolateoatmeal pan main)
     ;; (checkchocolateoatmeal pan  side)
     ;; (checkchocolateoatmeal pan new3)
     ;; (checkfruitychocolateoatmeal bowl main)
     ;; (checkfruitychocolateoatmeal bowl  side)
     ;; (checkfruitychocolateoatmeal bowl new3)
     ;; (checkfruitychocolateoatmeal pan main)
     ;; (checkfruitychocolateoatmeal pan  side)
     ;; (checkfruitychocolateoatmeal pan new3)
     ;; (checkpeanutbutterbananaoatmeal bowl main)
     ;; (checkpeanutbutterbananaoatmeal bowl  side)
     ;; (checkpeanutbutterbananaoatmeal bowl new3)
     ;; (checkpeanutbutterbananaoatmeal pan main)
     ;; (checkpeanutbutterbananaoatmeal pan  side)
     ;; (checkpeanutbutterbananaoatmeal pan new3)
     ;; (checknuttyfruityoatmeal bowl main)
     ;; (checknuttyfruityoatmeal bowl  side)
     ;; (checknuttyfruityoatmeal bowl new3)
     ;; (checknuttyfruityoatmeal pan main)
     ;; (checknuttyfruityoatmeal pan  side)
     ;; (checknuttyfruityoatmeal pan new3)
     ;; (checknuttychocolateoatmeal bowl main)
     ;; (checknuttychocolateoatmeal bowl  side)
     ;; (checknuttychocolateoatmeal bowl new3)
     ;; (checknuttychocolateoatmeal pan main)
     ;; (checknuttychocolateoatmeal pan  side)
     ;; (checknuttychocolateoatmeal pan new3)
     ;; (checknuttyfruitychocolateoatmeal bowl main)
     ;; (checknuttyfruitychocolateoatmeal bowl  side)
     ;; (checknuttyfruitychocolateoatmeal bowl new3)
     ;; (checknuttyfruitychocolateoatmeal pan main)
     ;; (checknuttyfruitychocolateoatmeal pan  side)
     ;; (checknuttyfruitychocolateoatmeal pan new3)
     ;; (checknuttypeanutbutterbananaoatmeal bowl main)
     ;; (checknuttypeanutbutterbananaoatmeal bowl  side)
     ;; (checknuttypeanutbutterbananaoatmeal bowl new3)
     ;; (checknuttypeanutbutterbananaoatmeal pan main)
     ;; (checknuttypeanutbutterbananaoatmeal pan  side)
     ;; (checknuttypeanutbutterbananaoatmeal pan new3)
     ;; (checknuttypeanutbutterchocolateoatmeal bowl main)
     ;; (checknuttypeanutbutterchocolateoatmeal bowl  side)
     ;; (checknuttypeanutbutterchocolateoatmeal bowl new3)
     ;; (checknuttypeanutbutterchocolateoatmeal pan main)
     ;; (checknuttypeanutbutterchocolateoatmeal pan  side)
     ;; (checknuttypeanutbutterchocolateoatmeal pan new3)


     ;; (turnon microwave)
     ;; (turnon stove)
     ;; (checkpie main)
     ;; (checkpie  side)
     ;; (checkpie new3)
     ;; (checkmuffin main)
     ;; (checkmuffin  side)
     ;; (checkmuffin new3)
     ;; (checkroll main)
     ;; (checkroll  side)
     ;; (checkroll new3)
     ;; (checkjellypastry main)
     ;; (checkjellypastry  side)
     ;; (checkjellypastry new3)
     ;; (checkegg main)
     ;; (checkegg  side)
     ;; (checkegg new3)


     ;; (takeout pie microwave)
     ;; (takeout muffin microwave)
     ;; (releasetool mixingspoon)
     ;; (collectwater measuringcup main)
     ;; (collectwater measuringcup  side)
     ;; (collectwater measuringcup new3)
     ;; (collectwater measuringcup water)
     
     ;; (boilliquid pan water)
     ;; (boilliquid measuringcup water)
     ;; (boilliquid bowl water)
     ;; (boilliquid pan milk)
     ;; (boilliquid measuringcup milk)
     ;; (boilliquid bowl milk)

     ;; (cookoatmeal pan bowl main)
     ;; (cookoatmeal pan bowl  side)
     ;; (cookoatmeal pan bowl new3)
     ;; (serveoatmeal pan bowl main)
     ;; (serveoatmeal pan bowl  side)
     ;; (serveoatmeal pan bowl  side)
     ;; (microwavepastry pie)
     ;; (microwavepastry muffin)
     ;; (microwavepastry roll)
     ;; (microwavepastry jellypastry)
     ;; (microwavepastry egg)


    )
    ;; (:goal (and (ncontainer water pan)
    ;;             (incontainer salt pan)
    ;;             (incontainer oatmeal pan)
    ;;             (mixed pan))
    ;;        )
    ;; (:goal (and
            ;; (simmering main)
            ;; (isoatmeal main)
            ;; (isfruityoatmeal main)

                ;; (mixed main)
                ;; )
           ;; )
     (:goal (and
             ;; (isnuttypeanutbutterchocolateoatmeal main)
             ;; (not (incontainer milk bowl))
             ;; (not (incontainer cocopuffs bowl))
             ;; (instorage pie)
             ;; (not (inworkspace milk))
             ;; (iscereal main)
             ;; (incontainer water measuringcup)
             ;; (inworkspace bowl)
             ;; (inworkspace milk)
             ;; (inworkspace cocopuffs)
             ;; (incontainer milk bowl)
             (iscompletedcereal main)
             ;; (isoatmeal main)
             (iscompletedmuffin  side)
             (iscompletedpastry  side)
                 )
            )
    )
